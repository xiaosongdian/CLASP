#!/usr/bin/env python3
"""
SFT Data Generator with Real Profile Generation
Generate real user profiles using GPT, replacing simple UserID placeholders
"""
import json
import random
import psycopg2
import requests
import os
import sys
import threading
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt import format_prompt, SYSTEM_INSTRUCTION_FREE

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 11860,
    'dbname': 'user_actions_db',
    'user': 'postgres',
    'password': 'Nudt_security@508!'
}

# API Configuration
os.environ['OPENAI_API_KEY'] = 'sk-1JY7edl1HTvrqYHQM8wfFSL72eNuhPJqM6WLJNNbqIciTUAB'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-1JY7edl1HTvrqYHQM8wfFSL72eNuhPJqM6WLJNNbqIciTUAB')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.huiyan-ai.cn/v1')
PROFILE_MODEL = 'gpt-4o-mini'  # LLM model for profile generation

T = 10
STEP = 1
TEXT_LONG = 500
# Profile generation: use at most N recent actions as FREE_FORM_PROMPT input
MAX_ACTIONS_FOR_PROFILE = 100
# SFT sample construction: use at most N recent actions per user (sliding window on this sequence)
MAX_ACTIONS_FOR_SFT = 40
PROFILE_CACHE_FILE = '/home/xiaosong/personality/BlueSky/SFT/profile_cache.json'
PROFILE_DB_FLUSH_EVERY = 10
PROFILE_THREADS = 8
API_TIMEOUT = 60          # Single API call timeout (seconds)
API_MAX_RETRIES = 2       # Retry count on failure (excluding first attempt)
HEARTBEAT_INTERVAL = 15   # Progress heartbeat interval (seconds), prevent "stuck" with no output
DB_FETCH_BATCH = 100000   # Server-side cursor batch fetch size (rows)
API_JITTER_MIN = 1.0      # Random jitter lower bound before each API call (seconds), avoid rate limiting
API_JITTER_MAX = 3.0      # Random jitter upper bound before each API call (seconds)

_cache_lock = threading.Lock()


def build_cache_key(user_id, community_id=None):
    """Build cache key from user_id and community_id; handles both int and str types."""
    if community_id is None:
        return str(user_id)
    return f"{community_id}:{user_id}"


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """
    Establish DB connection.
    - Enable TCP keepalive to prevent silent disconnection by intermediate network devices
    - Set statement_timeout=60s to prevent server hang and infinite client wait
    """
    return psycopg2.connect(
        **DB_CONFIG,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
        connect_timeout=10,
        options='-c statement_timeout=60000',
    )


def ensure_user_profile_cache_table(conn):
    """Create/ensure user_profile_cache table exists with all required columns."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile_cache (
            community_id INTEGER NOT NULL,
            user_id BIGINT NOT NULL,
            profile_text TEXT NOT NULL,
            sample_size INTEGER NOT NULL DEFAULT 0,
            profile_tokens INTEGER NOT NULL DEFAULT 0,
            profile_chars INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (community_id, user_id)
        )
        """
    )
    # Add columns if they don't exist (idempotent)
    cur.execute(
        """
        ALTER TABLE user_profile_cache
            ADD COLUMN IF NOT EXISTS sample_size INTEGER NOT NULL DEFAULT 0
        """
    )
    cur.execute(
        """
        ALTER TABLE user_profile_cache
            ADD COLUMN IF NOT EXISTS profile_tokens INTEGER NOT NULL DEFAULT 0
        """
    )
    cur.execute(
        """
        ALTER TABLE user_profile_cache
            ADD COLUMN IF NOT EXISTS profile_chars INTEGER NOT NULL DEFAULT 0
        """
    )
    cur.execute(
        """
        ALTER TABLE user_profile_cache
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        """
    )
    # Add primary key constraint if not exists (idempotent, skip if already exists)
    try:
        cur.execute(
            """
            ALTER TABLE user_profile_cache
                ADD CONSTRAINT user_profile_cache_pkey
                PRIMARY KEY (community_id, user_id)
            """
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  [ensure_table] ADD PRIMARY KEY skipped: {e}")
    # Create unique index if not exists (idempotent, skip if already exists)
    try:
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS user_profile_cache_uq
                ON user_profile_cache (community_id, user_id)
            """
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  [ensure_table] CREATE UNIQUE INDEX skipped: {e}")


def load_profiles_from_db(conn, community_id):
    """Load cached user profiles from user_profile_cache table (to avoid re-calling API)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id, profile_text
        FROM user_profile_cache
        WHERE community_id = %s
        """,
        (community_id,),
    )
    out = {}
    for user_id, profile_text in cur.fetchall():
        out[build_cache_key(user_id, community_id)] = profile_text
    return out


def flush_new_profiles_to_db(conn, community_id, batch):
    """
    Write a batch of newly generated profiles to user_profile_cache.
    Does not rely on unique constraints: uses INSERT ... SELECT ... WHERE NOT EXISTS,
    correctly inserts regardless of whether table has PK/UNIQUE constraints and doesn't overwrite existing rows.
    batch: list of (user_id, profile_text, sample_size, profile_tokens, profile_chars)
    """
    if not batch:
        return 0, 0
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for user_id, profile_text, actions_used, profile_tokens, profile_chars in batch:
        cur.execute(
            """
            INSERT INTO user_profile_cache
                (community_id, user_id, profile_text, sample_size, profile_tokens, profile_chars)
            SELECT %s, %s, %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM user_profile_cache
                WHERE community_id = %s AND user_id = %s
            )
            """,
            (community_id, user_id, profile_text, actions_used, profile_tokens, profile_chars,
             community_id, user_id),
        )
        if cur.rowcount == 1:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    return inserted, skipped


def apply_action_limits(user_actions):
    """
    Extract from complete time series:
    - profile: at most MAX_ACTIONS_FOR_PROFILE recent actions per user (for profile generation)
    - sft: at most MAX_ACTIONS_FOR_SFT recent actions per user (for SFT sample construction)
    """
    profile_map = {}
    sft_map = {}
    for uid, acts in user_actions.items():
        if len(acts) > MAX_ACTIONS_FOR_PROFILE:
            p = acts[-MAX_ACTIONS_FOR_PROFILE:]
        else:
            p = acts
        profile_map[uid] = p
        if len(p) > MAX_ACTIONS_FOR_SFT:
            sft_map[uid] = p[-MAX_ACTIONS_FOR_SFT:]
        else:
            sft_map[uid] = p
    return profile_map, sft_map


# ============================================================================
# PROFILE GENERATION FUNCTIONS
# ============================================================================

def load_profile_cache():
    """Load profile cache from JSON file."""
    if os.path.exists(PROFILE_CACHE_FILE):
        with open(PROFILE_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_profile_cache(cache):
    """Save profile cache to JSON file: write to tmp file first, then rename for atomicity."""
    # Take snapshot under lock to ensure consistency
    with _cache_lock:
        cache_snapshot = dict(cache)
    tmp_path = PROFILE_CACHE_FILE + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(cache_snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, PROFILE_CACHE_FILE)


def format_actions_for_profile(actions, max_len=MAX_ACTIONS_FOR_PROFILE):
    """Format action history as readable text for profile generation."""
    # Use most recent max_len actions
    recent_actions = actions[-max_len:] if len(actions) > max_len else actions
    
    lines = []
    for a in recent_actions:
        ts = a.get('timestamp', '')
        action_type = a.get('action_type', '')
        target = a.get('target', '')
        action_text = a.get('action_text', '') or ''
        original_text = a.get('original_text', '') or ''
        
        if action_type == 'reply':
            context = original_text[:200] if original_text else target[:200]
            content = action_text[:200]
            line = f"[{ts}] User commented on \"{context}\": \"{content}\""
        elif action_type == 'post':
            content = action_text[:200] if action_text else target[:200]
            line = f"[{ts}] User posted: \"{content}\""
        elif action_type == 'like':
            line = f"[{ts}] User liked: \"{target[:200]}\""
        elif action_type == 'repost':
            line = f"[{ts}] User reposted: \"{target[:200]}\""
        else:
            line = f"[{ts}] {action_type}: {target[:200]}"
        
        lines.append(line)
    
    return '\n'.join(lines)


def estimate_profile_tokens(profile_text):
    """
    Estimate tokens (only used when API doesn't return usage.completion_tokens).
    Rules: English words/numbers count as 1, CJK continuous character blocks count by character count, other symbols count as 1.
    """
    if not profile_text:
        return 0
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", profile_text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", profile_text)
    punct_or_other = re.findall(r"[^\sA-Za-z0-9_\u4e00-\u9fff]", profile_text)
    return len(ascii_tokens) + len(cjk_chars) + len(punct_or_other)

def _generate_profile_api_call(user_id, actions, community_id):
    """
    Call LLM to generate profile. Returns (profile_text, success_flag, profile_tokens).
    - On timeout or 5xx errors, retry with exponential backoff up to API_MAX_RETRIES times.
    - On all failures, return (fallback, False); caller won't cache fallback, can retry next time.
    """
    behavior_data = format_actions_for_profile(actions, MAX_ACTIONS_FOR_PROFILE)
    action_count = min(len(actions), MAX_ACTIONS_FOR_PROFILE)
    prompt = format_prompt(behavior_data=behavior_data, free_form=True, action_count=action_count)

    api_url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': PROFILE_MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_INSTRUCTION_FREE},
            {'role': 'user', 'content': prompt},
        ],
        'max_tokens': 2000,
    }

    last_err = None
    for attempt in range(API_MAX_RETRIES + 1):
        # Add 1-3s random jitter before each API call to avoid rate limiting
        jitter = random.uniform(API_JITTER_MIN, API_JITTER_MAX)
        time.sleep(jitter)
        try:
            resp = requests.post(
                api_url, headers=headers, json=payload,
                timeout=(10, API_TIMEOUT),  # (connect_timeout, read_timeout)
            )
            if resp.status_code == 200:
                result = resp.json()
                profile = result['choices'][0]['message']['content']
                usage = result.get('usage', {}) if isinstance(result, dict) else {}
                completion_tokens = usage.get('completion_tokens')
                if completion_tokens is None:
                    completion_tokens = estimate_profile_tokens(profile)
                return profile, True, int(completion_tokens)
            # 4xx errors (don't retry); 5xx/429 errors (retry)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                print(f"  [API 4xx] User {user_id}: {resp.status_code} - {resp.text[:120]}")
                return f"UserID {user_id} - Behavior: {len(actions)} actions", False, 0
            last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_err = str(e)
        if attempt < API_MAX_RETRIES:
            backoff = 2 ** attempt  # 1s, 2s
            print(
                f"  [API retry] User {user_id} attempt {attempt+1}/{API_MAX_RETRIES} "
                f"after {backoff}s; last={last_err}",
                flush=True,
            )
            time.sleep(backoff)
    print(f"  [API FAILED] User {user_id}: {last_err}", flush=True)
    return f"UserID {user_id} - Behavior: {len(actions)} actions", False, 0


def generate_user_profile(user_id, actions, cache, community_id=None):
    """
    Generate profile for a single user (with caching, thread-safe).
    Returns (profile_text, is_new, api_success, profile_tokens, profile_chars)
      - is_new: whether newly generated this time (not cache hit)
      - api_success: True only if API truly succeeded; False means fallback, shouldn't be cached
    """
    cache_key = build_cache_key(user_id, community_id)
    legacy_key = str(user_id)

    with _cache_lock:
        if cache_key in cache:
            profile = cache[cache_key]
            return profile, False, True, estimate_profile_tokens(profile), len(profile)
        if legacy_key in cache:
            cache[cache_key] = cache[legacy_key]
            profile = cache[cache_key]
            return profile, False, True, estimate_profile_tokens(profile), len(profile)

    profile, api_success, profile_tokens = _generate_profile_api_call(user_id, actions, community_id)
    profile_chars = len(profile)

    if api_success:
        with _cache_lock:
            cache[cache_key] = profile

    return profile, True, api_success, profile_tokens, profile_chars


def _generate_profile_worker(args):
    uid, actions, cache, community_id = args
    profile, is_new, api_success, profile_tokens, profile_chars = generate_user_profile(
        uid, actions, cache, community_id=community_id
    )
    actions_used = min(len(actions), MAX_ACTIONS_FOR_PROFILE)
    return uid, profile, actions_used, is_new, api_success, profile_tokens, profile_chars


def _flush_batch(_conn_holder_unused, community_id, pending_db, cache, stage_label=""):
    """
    Write pending_db to database and sync save JSON cache. Clear pending_db.

    Key: each flush uses a **fresh short-lived connection**, closes immediately after write.
    Avoids main thread idle connection being silently dropped, causing INSERT to hang forever.
    """
    if not pending_db:
        return
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        attempts += 1
        conn = None
        try:
            t0 = time.time()
            conn = get_db_connection()
            ins, sk = flush_new_profiles_to_db(conn, community_id, pending_db)
            print(
                f"  [DB flush{stage_label}] batch={len(pending_db)} "
                f"inserted={ins} skipped_existing={sk} "
                f"took={time.time() - t0:.2f}s",
                flush=True,
            )
            break
        except Exception as e:
            print(
                f"  [DB flush error] attempt {attempts}/{max_attempts}: {e}",
                flush=True,
            )
            if attempts >= max_attempts:
                print(
                    f"  [DB flush FAILED] giving up this batch of {len(pending_db)}; "
                    "cached in JSON only, retry on next run.",
                    flush=True,
                )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    save_profile_cache(cache)
    pending_db.clear()


def generate_all_profiles(user_actions, community_id):
    """
    Generate profiles for all users; use thread pool for concurrent API calls,
    flush to user_profile_cache and sync JSON every PROFILE_DB_FLUSH_EVERY new profiles.
    """
    print(f"\n=== Generating profiles for {len(user_actions)} users ===")

    cache = load_profile_cache()

    # Use list as holder for connection, _flush_batch will use it
    conn_holder = [get_db_connection()]
    ensure_user_profile_cache_table(conn_holder[0])
    db_map = load_profiles_from_db(conn_holder[0], community_id)
    for k, v in db_map.items():
        cache[k] = v

    users_to_generate = []
    for uid, actions in user_actions.items():
        ck = build_cache_key(uid, community_id)
        if ck not in cache and str(uid) not in cache:
            users_to_generate.append((uid, actions, cache, community_id))

    print(
        f"Cache (json+DB): {len(cache)} entries, "
        f"DB rows this community: {len(db_map)}, Need to generate: {len(users_to_generate)}"
    )

    if not users_to_generate:
        try:
            conn_holder[0].close()
        except Exception:
            pass
        return cache

    print(
        f"[ThreadPool] Starting {len(users_to_generate)} users "
        f"with {PROFILE_THREADS} workers (flush every {PROFILE_DB_FLUSH_EVERY})...",
        flush=True,
    )

    pending_db = []
    new_success = 0
    new_failed = 0
    done = 0
    total = len(users_to_generate)
    start_ts = time.time()

    # ---------- Heartbeat thread: monitor progress, print status ----------
    stop_heartbeat = threading.Event()

    def _heartbeat():
        while not stop_heartbeat.wait(HEARTBEAT_INTERVAL):
            elapsed = time.time() - start_ts
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_str = f"{(total - done) / rate:.0f}s" if rate > 0 else "?"
            print(
                f"  [Heartbeat] {done}/{total} done "
                f"(new_success={new_success}, new_failed={new_failed}, "
                f"pending_flush={len(pending_db)}) "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta_str}",
                flush=True,
            )

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=PROFILE_THREADS) as executor:
            futures = {
                executor.submit(_generate_profile_worker, args): args[0]
                for args in users_to_generate
            }
            for future in as_completed(futures):
                uid = futures[future]
                done += 1
                try:
                    uid_ret, profile, actions_used, is_new, api_success, profile_tokens, profile_chars = future.result()
                except Exception as e:
                    print(f"  [Worker Error] User {uid}: {e}", flush=True)
                    continue

                if is_new and api_success:
                    pending_db.append((uid_ret, profile, actions_used, profile_tokens, profile_chars))
                    new_success += 1
                elif is_new and not api_success:
                    new_failed += 1

                if done % 5 == 0 or done == total:
                    print(
                        f"  Progress: {done}/{total} "
                        f"(new_success={new_success}, new_failed={new_failed}, "
                        f"pending_flush={len(pending_db)})",
                        flush=True,
                    )

                if len(pending_db) >= PROFILE_DB_FLUSH_EVERY:
                    _flush_batch(conn_holder, community_id, pending_db, cache)

        _flush_batch(conn_holder, community_id, pending_db, cache, stage_label=" final")
    finally:
        stop_heartbeat.set()
        try:
            if conn_holder[0] is not None:
                conn_holder[0].close()
        except Exception:
            pass

    print(
        f"[ThreadPool] Completed. new_success={new_success}, "
        f"new_failed(api_fallback, not cached)={new_failed}"
    )
    print(f"Profile cache saved to {PROFILE_CACHE_FILE}")
    return cache

# ============================================================================
# SAMPLE CREATION FUNCTIONS
# ============================================================================

def format_action(a):
    ts = a.get('timestamp', '')
    action_type = a.get('action_type', '')
    target = a.get('target', '')
    action_text = a.get('action_text', '')
    original_text = a.get('original_text', '')
    
    if action_type == 'reply':
        context = original_text[:TEXT_LONG] if original_text else target[:TEXT_LONG]
        content = action_text[:TEXT_LONG] if action_text else ''
        return f"[{ts}] User commented on \"{context}\": \"{content}\""
    elif action_type == 'post':
        content = action_text[:TEXT_LONG] if action_text else target[:TEXT_LONG]
        return f"[{ts}] User posted: \"{content}\""
    elif action_type == 'like':
        return f"[{ts}] User liked: \"{target[:TEXT_LONG]}...\""
    elif action_type == 'repost':
        return f"[{ts}] User reposted: \"{target[:TEXT_LONG]}...\""
    else:
        return f"[{ts}] User performed {action_type} on: \"{target[:TEXT_LONG]}...\""

def create_content_sample(user_profile, history, target_action):
    history_str = "\n".join([format_action(a) for a in history])
    action_type = target_action['action_type']
    action_text = target_action.get('action_text', '')
    target = target_action.get('target', '')
    original_text = target_action.get('original_text', '')
    
    if action_type == 'post':
        output = action_text if action_text else target
        scenario = "Write a post, what content?"
    elif action_type == 'reply':
        output = action_text if action_text else ''
        context = original_text[:TEXT_LONG] if original_text else target[:TEXT_LONG]
        scenario = f"Comment on \"{context}...\", what is the comment content?"

    return {
        "Instruction": "Role: Social media user behavior simulation expert\nCore requirement: Match target user profile and historical behavior sequence, output content fully consistent with user characteristics\nTask objective: Restore the real content the user would publish in the specified scenario\nOutput rule: Only output the content generated by the user, no additional explanation",
        "input": f"Target user profile: {user_profile}\nUser historical behavior sequence: {history_str}\nCurrent scenario: After completing the above historical behaviors, {scenario}",
        "output": output
    }

def create_decision_sample(user_profile, history, target_action):
    history_str = "\n".join([format_action(a) for a in history])
    action_type = target_action['action_type']
    target = target_action.get('target', '')
    original_text = target_action.get('original_text', '')
    if len(target) > TEXT_LONG:
        target = target[:TEXT_LONG]+"..."
    if original_text and len(original_text) > TEXT_LONG:
        original_text = original_text[:TEXT_LONG]+"..."
    available = "like, repost, reply, post, not interested"
    
    if action_type == 'post':
        scenario = f"Given that the user's behavior contains the following content: {target}\nWhich type of user behavior is this most likely generated from?"
    elif action_type == 'reply':
        context = original_text if original_text else target
        scenario = f"Given that the user's behavior contains the following content: {context}\nWhich type of user behavior is this most likely generated from?"
    elif action_type == 'repost':
        context = original_text if original_text else target
        scenario = f"Given that the user's behavior contains the following content: {context}\nWhich type of user behavior is this most likely generated from?"
    elif action_type == 'like':
        context = original_text if original_text else target
        scenario = f"Given that the user's behavior contains the following content: {context}\nWhich type of user behavior is this most likely generated from?"
    else:
        scenario = "What action to take"
    
    return {
        "Instruction": f"Role: Social media user behavior simulation expert\nCore requirement: Match target user profile and historical behavior sequence, output interaction decision fully consistent with user characteristics\nTask objective: Predict what action the user will take next based on historical behavior\nOutput rule: Select one from the available actions",
        "input": f"Target user profile: {user_profile}\nUser historical behavior sequence: {history_str}\nCurrent scenario: After completing the above historical behaviors, {scenario}\nAvailable actions: {available}",
        "output": action_type
    }

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def _row_to_action(r):
    """Convert a SELECT result row to user_id and action dict."""
    uid = r[0]
    action_type = r[1]
    date = str(r[6])
    if action_type == 'reply':
        target = str(r[5])[:TEXT_LONG] if r[5] else (str(r[4])[:TEXT_LONG] if r[4] else f"post_{r[2]}")
    else:
        target = str(r[4])[:TEXT_LONG] if r[4] else (str(r[5])[:TEXT_LONG] if r[5] else f"post_{r[2]}")
    year, month, day = date[0:4], date[4:6], date[6:8]
    hour, minute = date[8:10], date[10:12]
    return uid, {
        'timestamp': f"{year}-{month}-{day} {hour}:{minute}",
        'action_type': action_type,
        'target': target,
        'action_text': r[4],
        'original_text': r[5],
        'date': date,
    }


def _stream_user_actions(community_id, selected_user_ids=None):
    """
    Stream fetch using server-side cursor, print progress every DB_FETCH_BATCH rows.
    Returns {user_id: [action, ...]} dictionary.
    """
    conn = get_db_connection()
    # Use server-side cursor with itersize for streaming
    cur = conn.cursor(name=f"sft_stream_{community_id}_{os.getpid()}")
    cur.itersize = DB_FETCH_BATCH

    if selected_user_ids is not None:
        sql = """
            SELECT s.user_id, a.action_type, a.post_id, a.target_post_id,
                   a.action_text, a.original_text, a.date
            FROM user_actions_sampled a
            JOIN user_stats s ON a.user_id = s.user_id
            WHERE s.community_id = %s AND s.user_id = ANY(%s)
            ORDER BY a.user_id, a.date
        """
        params = (community_id, selected_user_ids)
    else:
        sql = """
            SELECT s.user_id, a.action_type, a.post_id, a.target_post_id,
                   a.action_text, a.original_text, a.date
            FROM user_actions_sampled a
            JOIN user_stats s ON a.user_id = s.user_id
            WHERE s.community_id = %s
            ORDER BY a.user_id, a.date
        """
        params = (community_id,)

    print(f"[DB] Executing action query for community {community_id} ...", flush=True)
    t0 = time.time()
    cur.execute(sql, params)
    print(f"[DB] Query executed in {time.time() - t0:.1f}s, streaming rows...", flush=True)

    user_actions = {}
    total_rows = 0
    last_log_ts = time.time()
    while True:
        batch = cur.fetchmany(DB_FETCH_BATCH)
        if not batch:
            break
        for r in batch:
            uid, act = _row_to_action(r)
            if uid not in user_actions:
                user_actions[uid] = []
            user_actions[uid].append(act)
        total_rows += len(batch)
        now = time.time()
        if now - last_log_ts >= 5 or len(batch) < DB_FETCH_BATCH:
            print(
                f"  [DB stream] {total_rows} rows loaded, "
                f"users so far: {len(user_actions)}, "
                f"elapsed={now - t0:.1f}s",
                flush=True,
            )
            last_log_ts = now

    cur.close()
    conn.close()
    print(f"[DB] Finished: {total_rows} rows, {len(user_actions)} users", flush=True)
    return user_actions


def generate_sft_data(community_id, output_file, use_real_profile=True, max_users=None):
    print(f"Loading data for community {community_id}...")

    # If max_users is specified, randomly sample user IDs
    selected_user_ids = None
    if max_users is not None:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT s.user_id
            FROM user_stats s
            WHERE s.community_id = %s
            """,
            (community_id,),
        )
        all_user_ids = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        total_users = len(all_user_ids)
        print(f"Community {community_id} has {total_users} users total")
        if total_users > max_users:
            selected_user_ids = random.sample(all_user_ids, max_users)
            print(f"Randomly selected {max_users} users from {total_users}")
        else:
            selected_user_ids = all_user_ids
            print(f"Community only has {total_users} users, using all")

    user_actions = _stream_user_actions(community_id, selected_user_ids)

    user_actions_profile, user_actions_sft = apply_action_limits(user_actions)

    print(
        f"Users: {len(user_actions_profile)} "
        f"(profile<= {MAX_ACTIONS_FOR_PROFILE} actions, SFT<= {MAX_ACTIONS_FOR_SFT} actions per user)"
    )

    # Generate user profiles (using at most MAX_ACTIONS_FOR_PROFILE actions)
    profile_cache = {}
    if use_real_profile:
        profile_cache = generate_all_profiles(user_actions_profile, community_id)
    else:
        # Fallback: use simple placeholders
        for uid in user_actions_profile.keys():
            profile_cache[build_cache_key(uid, community_id)] = (
                f"UserID {uid}, Community {community_id}"
            )

    # Generate SFT samples (using at most MAX_ACTIONS_FOR_SFT actions with sliding window)
    samples = []

    for uid, actions in user_actions_sft.items():
        n = len(actions)
        if n <= T:
            continue

        cache_key = build_cache_key(uid, community_id)
        user_profile = profile_cache.get(cache_key, profile_cache.get(str(uid), f"UserID {uid}, Community {community_id}"))

        for start in range(0, n - T, STEP):
            history = actions[start:start + T]
            target = actions[start + T]

            at = target['action_type']
            if at in ['post', 'reply'] and random.random() < 0.3:
                sample = create_decision_sample(user_profile, history, target)
            elif at in ['post', 'reply']:
                sample = create_content_sample(user_profile, history, target)
            else:
                sample = create_decision_sample(user_profile, history, target)

            # Add user_id for tracking
            sample['user_id'] = uid
            samples.append(sample)

    print(f"Samples generated: {len(samples)}")

    # Count sample types
    content_count = sum(1 for s in samples if 'Write a post' in s['input'] or 'Comment on' in s['input'])
    decision_count = len(samples) - content_count
    print(f"  - Content samples: {content_count}")
    print(f"  - Decision samples: {decision_count}")

    # Save complete dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"Saved to {output_file}")
    return samples

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def test_mode(community_id=9, output_file=None):
    """Test mode: process 2 users"""
    print(f"=== TEST MODE: Processing 2 users ===")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.user_id, COUNT(*) as cnt
        FROM user_actions_sampled a
        JOIN user_stats s ON a.user_id = s.user_id
        WHERE s.community_id = %s
        GROUP BY s.user_id
        HAVING COUNT(*) > 10
        LIMIT 2
    """, (community_id,))

    test_users = [row[0] for row in cursor.fetchall()]
    print(f"Test users: {test_users}")

    cursor.execute("""
        SELECT s.user_id, a.action_type, a.post_id, a.target_post_id, a.action_text, a.original_text, a.date
        FROM user_actions_sampled a
        JOIN user_stats s ON a.user_id = s.user_id
        WHERE s.community_id = %s AND s.user_id = ANY(%s)
        ORDER BY a.user_id, a.date
    """, (community_id, test_users))

    rows = cursor.fetchall()
    conn.close()
    print(f"Loaded {len(rows)} records for test users")

    user_actions = {}
    for r in rows:
        uid = r[0]
        action_type = r[1]
        date = str(r[6])

        if action_type == 'reply':
            target = str(r[5])[:TEXT_LONG] if r[5] else (str(r[4])[:TEXT_LONG] if r[4] else f"post_{r[2]}")
        else:
            target = str(r[4])[:TEXT_LONG] if r[4] else (str(r[5])[:TEXT_LONG] if r[5] else f"post_{r[2]}")

        if uid not in user_actions:
            user_actions[uid] = []

        year = date[0:4]
        month = date[4:6]
        day = date[6:8]
        hour = date[8:10]
        minute = date[10:12]

        user_actions[uid].append({
            'timestamp': f"{year}-{month}-{day} {hour}:{minute}",
            'action_type': action_type,
            'target': target,
            'action_text': r[4],
            'original_text': r[5],
            'date': date
        })

    user_actions_profile, user_actions_sft = apply_action_limits(user_actions)

    print(f"Users: {len(user_actions_profile)}")

    for uid in user_actions_profile.keys():
        print(
            f"  User {uid}: profile_actions={len(user_actions_profile[uid])}, "
            f"sft_actions={len(user_actions_sft[uid])}"
        )

    print("\n=== Generating profiles ===")
    profile_cache = generate_all_profiles(user_actions_profile, community_id)

    print("\n=== Generating SFT samples ===")
    samples = []

    for uid, actions in user_actions_sft.items():
        n = len(actions)
        if n <= T:
            continue

        cache_key = build_cache_key(uid, community_id)
        user_profile = profile_cache.get(cache_key, profile_cache.get(str(uid), f"UserID {uid}, Community {community_id}"))

        for start in range(0, n - T, STEP):
            history = actions[start:start + T]
            target = actions[start + T]

            at = target['action_type']
            if at in ['post', 'reply'] and random.random() < 0.3:
                sample = create_decision_sample(user_profile, history, target)
            elif at in ['post', 'reply']:
                sample = create_content_sample(user_profile, history, target)
            else:
                sample = create_decision_sample(user_profile, history, target)

            sample['user_id'] = uid
            samples.append(sample)

    print(f"Total samples: {len(samples)}")

    content_count = sum(1 for s in samples if 'Write a post' in s['input'] or 'Comment on' in s['input'])
    decision_count = len(samples) - content_count
    print(f"  - Content: {content_count}")
    print(f"  - Decision: {decision_count}")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"Saved to {output_file}")

    print("\n=== Sample Examples ===")
    for i, s in enumerate(samples[:3]):
        sample_type = 'Content' if 'Write a post' in s['input'] or 'Comment on' in s['input'] else 'Decision'
        print(f"\n--- Sample {i+1} (User {s.get('user_id')}, {sample_type}) ---")
        print(f"Output: {s['output'][:150]}")
        print(f"Input: {s['input'][:300]}")

    return samples


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_mode(community_id=9, output_file='/home/xiaosong/personality/BlueSky/SFT/test_2users.jsonl')
    else:
        cid_arg = sys.argv[1] if len(sys.argv) > 1 else "9"
        out = sys.argv[2] if len(sys.argv) > 2 else f"/home/xiaosong/personality/BlueSky/SFT/sft_cids_{cid_arg.replace(',', '_')}_with_profile.jsonl"
        use_real = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        max_users = int(sys.argv[4]) if len(sys.argv) > 4 else None
        cids = [int(c.strip()) for c in cid_arg.split(",") if c.strip()]
        if not cids:
            raise ValueError("No valid community id provided")

        print(f"=== SFT Data Generation with Real Profiles ===")
        print(f"Communities: {cids}")
        print(f"Output: {out}")
        print(f"Use real profiles: {use_real == 1}")
        print(f"Max users: {max_users if max_users else 'All'}")
        print(f"Max actions for profile: {MAX_ACTIONS_FOR_PROFILE}, for SFT: {MAX_ACTIONS_FOR_SFT}")
        print("Profile prompt mode: FREE_FORM_PROMPT")
        print()
        all_samples = []
        for cid in cids:
            community_out = out
            if len(cids) > 1:
                community_out = out.replace(".jsonl", f".cid{cid}.jsonl")
            samples = generate_sft_data(cid, community_out, use_real_profile=(use_real == 1), max_users=max_users)
            all_samples.extend(samples)
        if len(cids) > 1:
            with open(out, 'w', encoding='utf-8') as f:
                for s in all_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
            print(f"Merged {len(all_samples)} samples to {out}")
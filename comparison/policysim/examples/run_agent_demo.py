from policysim.agent import PolicyAgent, AgentProfile
from policysim.config import ModelConfig
from policysim.llm import OpenAIChatModel


def main() -> None:
    profile = AgentProfile(
        user_id="user_001",
        attributes={
            "likely_identity": "college student",
            "interested_areas": "social issues, technology",
            "posting_style": "direct and short",
            "interaction_behavior": "active in replies",
        },
    )
    model = OpenAIChatModel(ModelConfig())
    agent = PolicyAgent(profile=profile, model=model)
    actions = agent.act(
        topic="platform governance",
        trigger_news="平台宣布将降低误导信息内容曝光。",
        incoming_message="我觉得平台不应该干预推荐，用户自己判断就好。",
    )
    for action in actions:
        print(action)


if __name__ == "__main__":
    main()


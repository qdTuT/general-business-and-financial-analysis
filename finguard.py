import os
import re
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import datetime

# 获取当前文件的父目录的父目录下的 .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))


# 1. Schema: 定义状态数据结构 (State)
class AgentState(TypedDict):
    task: str  # 用户输入的广泛商业/金融问题
    search_queries: List[str]  # AI 自动拆解出的搜索关键词
    raw_data: str  # 从全网抓取回来的非结构化数据
    draft_report: str  # 分析师写出的初稿
    feedback: List[str]  # 审查官给出的修改建议
    quality_score: int  # 质量与合规评分
    iteration_count: int  # 迭代次数


# 2. Tools: 外部工具集成 (搜索引擎)
search_tool = DuckDuckGoSearchResults(max_results=5)


def perform_web_research(query: str) -> str:
    """执行网络搜索并返回摘要文本"""
    try:
        results = search_tool.invoke(query)
        return results if results else "未检索到相关有效信息。"
    except Exception as e:
        return f"搜索过程中发生错误: {str(e)}"


# 3. Agents: 各个智能体的核心业务逻辑
llm = ChatOpenAI(
    model=os.getenv("AIHUBMIX_MODEL_ID"),
    openai_api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url=os.getenv("AIHUBMIX_BASE_URL"),
    temperature=0.2
)


def researcher_node(state: AgentState):
    """数据侦察员：引入动态时间锚点，强制执行 2026 年实时检索"""
    # 获取当前真实时间
    now = datetime.datetime.now()
    current_date_str = now.strftime("%Y年%m月%d日")
    current_year = str(now.year)

    print(f"\n[Scout Agent] 当前系统时间: {current_date_str}")

    # 设定最大尝试次数，防止死循环
    max_retries = 3
    queries = []

    for i in range(max_retries):
        print(f"[Scout Agent] 正在尝试生成实时搜索词 (第 {i + 1} 次尝试)...")
        prompt = ChatPromptTemplate.from_template(
            "【当前时间锚点】：{current_date}\n"
            "【任务】：分析用户请求 '{task}'。\n"
            "【硬性约束】：\n"
            "1. 你必须基于 {current_year} 年及以后的视角生成关键词。\n"
            "2. 严禁使用任何早于 {current_year} 年的年份。如果涉及节假日分析，必须指明是 {current_year} 年的该节假日。\n"
            "3. 建议包含：'最新消息', '{current_year}年走势', '行业研报', '政策解读' 等词。\n\n"
            "请给出3个最能获取【最新】和【前瞻】信息的搜索关键词(用英文逗号分隔，无废话)。"
        )

        query_chain = prompt | llm
        # 注入当前时间和年份
        res = query_chain.invoke({
            "task": state['task'],
            "current_date": current_date_str,
            "current_year": current_year
        })

        temp_queries = [q.strip().strip('"') for q in res.content.split(',')]

        if all(current_year in q for q in temp_queries):
            queries = temp_queries
            break
        else:
            if i == max_retries - 1:
                # 最后一次尝试若仍失败，执行兜底：强制给每个词打上年份标签
                queries = [f"{q} {current_year}" if current_year not in q else q for q in temp_queries]

    all_data = []
    for q in queries:
        print(f"  -> 🔍 执行严格实时检索: {q}")
        search_result = perform_web_research(q)
        all_data.append(f"【数据源: {q}】\n{search_result}")

    return {
        "search_queries": queries,
        "raw_data": "\n\n".join(all_data),
        "iteration_count": 0
    }


def analyst_node(state: AgentState):
    """业务分析师：负责根据生肉数据撰写研报草案"""
    current_iter = state.get('iteration_count', 0) + 1
    now_year = datetime.datetime.now().year
    print(f"\n[Analyst Agent] 正在撰写深度报告 (当前年份参考: {now_year}年)...")

    prompt = ChatPromptTemplate.from_template(
        "你是一位顶级的金融分析师。请处理以下数据并撰写报告。\n"
        "【重要指令】：\n"
        "1. 你的分析必须完全基于 {now_year} 年的现实情况。如果搜索数据中混入了往年的过时信息，请识别并剔除，不得采纳。\n"
        "2. 深度结合基本面、消息面、政策面。如果是节假日后分析，重点关注最新消费数据及后续政策导向。\n\n"
        "参考数据: {data}\n"
        "任务目标: {task}"
        "前序审查意见: {feedback}\n\n"
        "要求：使用 Markdown 格式，包含核心洞察与未来趋势预测。"
    )

    chain = prompt | llm
    res = chain.invoke({
        "now_year": now_year,
        "task": state['task'],
        "data": state['raw_data'],
        "feedback": "\n".join(state.get('feedback', [])) or "无，这是初稿。"
    })

    return {"draft_report": res.content, "iteration_count": current_iter}


def reviewer_node(state: AgentState):
    """合规审查官：负责逻辑与合规性审查并打分"""
    print("\n[Reviewer Agent] 正在进行严密的逻辑与合规审查...")

    prompt = ChatPromptTemplate.from_template(
        "你是一家顶级投行的首席风控与合规官。请审查以下报告草案。\n"
        "审查标准：\n"
        "1. 数据支撑：结论是否有事实依据？\n"
        "2. 客观性：是否过度乐观或悲观？\n"
        "3. 风险提示：是否充分揭示了潜在的商业/政策风险？\n\n"
        "报告草案: {report}\n\n"
        "请给出你的审查意见，并在最后一行严格按此格式打分(0-100)：'最终评分: 85'"
    )
    chain = prompt | llm
    res = chain.invoke({"report": state['draft_report']})
    review_text = res.content

    # 提取分数
    score_match = re.search(r"最终评分:\s*(\d+)", review_text)
    score = int(score_match.group(1)) if score_match else 70  # 解析失败则给个低分强制重写

    print(f"  -> 审查完毕，当前报告得分: {score}")
    return {"quality_score": score, "feedback": [review_text]}


# 4. Graph: 构建多 Agent 协作工作流 (StateGraph)
def build_workflow():
    """编排工作流节点与流转逻辑"""
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reviewer", reviewer_node)

    # 设定主干连线
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "reviewer")

    # 设定条件分流逻辑 (长链推理的核心)
    def should_continue(state: AgentState):
        if state["quality_score"] < 85 and state["iteration_count"] < 3:
            print("  ==> 质量未达标，打回给分析师重写 <==")
            return "analyst"
        print("  ==> 质量达标或达到最大迭代次数，流程结束 <==")
        return END

    workflow.add_conditional_edges("reviewer", should_continue)

    return workflow.compile()


# 5. Main: 交互式运行入口
def generate_dynamic_examples():
    """在启动前，自动抓取今日热点并生成推荐提问"""
    print("⏳ 正在全网扫描今日财经热点，生成动态示例...\n")
    try:
        # 搜索今日宏观或热门板块新闻
        hot_news = perform_web_research("今日 股市 板块 核心热点 财经新闻")

        prompt = ChatPromptTemplate.from_template(
            "你是一个敏锐的金融主编。请根据以下刚刚抓取到的最新财经新闻，生成 3 个非常有深度的、适合让AI Agent分析的商业/金融课题。\n"
            "参考新闻：{news}\n\n"
            "要求：\n"
            "1. 必须是具体的公司新闻或具体的板块趋势分析。\n"
            "2. 侧重于当前影响和未来预测。\n"
            "3. 只输出3个用 '-' 开头的条目，不要任何问候语和废话。\n"
            "格式示例：\n"
            " - 分析XX公司最新发布的XX产品对行业的颠覆性及未来市占率预测\n"
        )
        chain = prompt | llm
        examples = chain.invoke({"news": hot_news}).content
        return examples
    except Exception as e:
        # 降级方案：如果搜索失败，使用默认示例
        return (" --- 动态示例生成有误，以下提供默认示例 ---\n"
                " - 分析英伟达最新财季指引对全球AI算力板块未来半年的估值影响\n"
                " - 评估固态电池最新技术突破对传统锂电产业链的冲击与洗牌\n"
                " - 预测美联储最新利率决议对出海中东电商企业的汇率与融资成本影响")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 欢迎使用 FinGuard 泛商业与金融分析 Agent 集群")
    print("=" * 70)

    # 动态生成今日示例
    dynamic_examples = generate_dynamic_examples()
    print("💡 【今日实时 AI 推荐分析课题】：")
    print(dynamic_examples)
    print("-" * 70 + "\n")

    # 构建图
    app = build_workflow()

    while True:
        try:
            user_input = input("👉 请输入你需要 Agent 分析的课题 (可参考上方提示，输入 q 退出): \n> ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("感谢使用，再见！")
                break

            if not user_input.strip():
                continue

            initial_state = {
                "task": user_input,
                "iteration_count": 0,
                "feedback": [],
                "quality_score": 0
            }

            final_state = app.invoke(initial_state)

            print("\n\n" + "★" * 25 + " 最终输出报告 " + "★" * 25)
            print(final_state['draft_report'])
            print("★" * 64 + "\n")

        except KeyboardInterrupt:
            print("\n已手动中断执行。")
            break
        except Exception as e:
            print(f"\n执行过程中发生错误: {e}")

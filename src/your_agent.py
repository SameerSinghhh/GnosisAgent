import autogen
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.gtypes import Probability
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import SecretStr
import time

# 🔹 Load API Keys
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BET_FROM_PRIVATE_KEY = os.getenv("BET_FROM_PRIVATE_KEY")

# 🔹 AutoGen Configuration
config_list = [{"model": "gpt-4o-mini-2024-07-18", "api_key": OPENAI_API_KEY}]

# 🔹 Define AutoGen Agents
question_expansion_agent = autogen.AssistantAgent(
    name="QuestionExpansionAgent",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message="You are a research question expansion expert."
)
news_retrieval_agent = autogen.AssistantAgent(
    name="NewsRetrievalAgent",
    llm_config={"config_list": config_list, "temperature": 0.5},
    system_message="You are a news retrieval expert."
)
scraper_agent = autogen.AssistantAgent(
    name="ScraperAgent",
    llm_config={"config_list": config_list, "temperature": 0.5},
    system_message="You are a web scraping expert."
)
analysis_agent = autogen.AssistantAgent(
    name="AnalysisAgent",
    llm_config={"config_list": config_list, "temperature": 0.3},
    system_message="You are a probability analysis expert."
)
betting_agent = autogen.AssistantAgent(
    name="BettingAgent",
    llm_config={"config_list": config_list, "temperature": 0.3},
    system_message="You are a betting strategy expert."
)

# 🔹 AI-Powered GPT-4o Mini Model
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    api_key=SecretStr(OPENAI_API_KEY),
    temperature=0.5,
)

# 🔹 Functions for Each Agent's Task

def generate_research_questions(market_question: str):
    """ Expands market question into 5 specific research questions. """
    print(f"🔍 [QuestionExpansionAgent] Expanding: {market_question}")
    
    prompt = f"""
    If I were to research this question to answer it fully, what aspects should I research?
    This is the question: {market_question}
    Make the research specific (e.g., instead of broad topics, make it time-bound or detailed).
    Provide exactly 5 different questions.
    """
    
    # Start a conversation with the agent
    response = question_expansion_agent.generate_reply(
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract the response from the AutoGen agent
    research_questions = response.split("\n")  # Assume list is returned as text
    research_questions = [q.strip() for q in research_questions if q.strip()]

    print(f"✅ [QuestionExpansionAgent] Research Questions: {research_questions}")
    return research_questions[:5]  # Ensure exactly 5 questions


def fetch_news_articles(research_questions: list[str]):
    """ Uses SERPER API to retrieve news links for each research question. """
    print(f"📰 [NewsRetrievalAgent] Fetching news for research questions.")
    
    all_articles = []
    for question in research_questions:
        print(f"🔎 Searching: {question}")
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": question},
        )
        # Filter out YouTube links and ensure we have valid links
        links = [
            x["link"] for x in response.json().get("organic", []) 
            if x.get("link") and not any(youtube in x["link"].lower() for youtube in ["youtube.com", "youtu.be"])
        ]
        print(f"✅ [NewsRetrievalAgent] Found {len(links)} valid articles for '{question}'")
        all_articles.extend(links[:3])  # Limit per question
    
    return all_articles

def scrape_article_content(url: str):
    """ Scrapes text content from an article using Firecrawl, Tavily, or BeautifulSoup fallback. """
    print(f"🌍 [ScraperAgent] Scraping article: {url}")

    def firecrawl():
        if not FIRECRAWL_API_KEY:
            print("⚠️ [Firecrawl] No API key provided")
            return None
        print(f"🔥 [Firecrawl] Scraping: {url}")
        try:
            response = requests.post(
                "https://api.firecrawl.com/v1/scrape",  # Updated endpoint
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"url": url, "formats": ["markdown"]},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("markdown", "")[:5000]
        except requests.exceptions.RequestException as e:
            print(f"❌ [Firecrawl] Error: {str(e)}")
            return None

    def tavily():
        if not TAVILY_API_KEY:
            print("⚠️ [Tavily] No API key provided")
            return None
        print(f"🔹 [Tavily] Scraping: {url}")
        try:
            response = requests.post(
                "https://api.tavily.com/v1/scrape",  # Updated endpoint
                headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
                json={"url": url},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("content", "")[:5000]
        except requests.exceptions.RequestException as e:
            print(f"❌ [Tavily] Error: {str(e)}")
            return None

    def beautifulsoup():
        print(f"🛠 [BeautifulSoup] Scraping fallback: {url}")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            # Get text content
            text = soup.get_text()
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text[:5000]
        except requests.exceptions.RequestException as e:
            print(f"❌ [BeautifulSoup] Error: {str(e)}")
            return None

    # Try each scraping method in order of preference
    content = None
    for scraper in [firecrawl, tavily, beautifulsoup]:
        content = scraper()
        if content:
            print(f"✅ [ScraperAgent] Successfully scraped content from {url}")
            break
    
    if not content:
        print(f"❌ [ScraperAgent] Failed to scrape content from {url}")
    return content

def analyze_articles(market_question: str, articles: list[str]):
    """ Uses AI to analyze article content and estimate probability. """
    print(f"📊 [AnalysisAgent] Analyzing articles for: {market_question}")

    # Filter out None values and empty strings
    valid_articles = [article for article in articles if article and article.strip()]
    
    if not valid_articles:
        print("❌ [AnalysisAgent] No valid articles to analyze")
        return 0.5, 0.5, "No valid articles to analyze"

    # Debug: Print the first few characters of each article
    print(f"📄 [AnalysisAgent] Found {len(valid_articles)} articles to analyze")
    for i, article in enumerate(valid_articles):
        print(f"Article {i+1} preview: {article[:100]}...")

    # Format articles with clear separation
    formatted_articles = []
    for i, article in enumerate(valid_articles):
        formatted_articles.append(f"Article {i+1}:\n{article}\n{'='*80}")

    # Create a more detailed prompt for the analysis agent
    prompt = f"""
    You are an expert at analyzing news articles and predicting event probabilities.
    
    Market Question: "{market_question}"
    
    Below are the relevant news articles to analyze:
    {'-' * 80}
    {'\n\n'.join(formatted_articles)}
    {'-' * 80}
    
    Based on the above articles, please:
    1. Analyze the content thoroughly
    2. Consider the credibility and recency of the information
    3. Evaluate any conflicting information
    4. Provide your assessment of the probability (0.0 to 1.0) that this event will happen
    5. Provide your confidence in this assessment (0.0 to 1.0)
    6. Provide a brief explanation (2-3 sentences) of your reasoning
    
    Return your response in this exact format:
    PROBABILITY CONFIDENCE REASONING
    Example: 0.75 0.85 Based on recent NASA announcements and successful test results, the probability is high. The information comes from reliable sources and shows clear progress.
    """
    
    # Debug: Print the prompt length
    print(f"📝 [AnalysisAgent] Prompt length: {len(prompt)} characters")

    response = analysis_agent.generate_reply(
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"✅ [AnalysisAgent] Raw Analysis Result: {response}")

    try:
        # Split response into parts
        parts = response.strip().split(' ', 2)
        if len(parts) >= 3:
            probability = float(parts[0])
            confidence = float(parts[1])
            reasoning = parts[2]
            
            # Ensure values are within valid ranges
            probability = max(0.0, min(1.0, probability))
            confidence = max(0.0, min(1.0, confidence))
            
            print(f"✅ [AnalysisAgent] Parsed Analysis Result: Probability={probability}, Confidence={confidence}")
            print(f"📝 [AnalysisAgent] Reasoning: {reasoning}")
            
            return probability, confidence, reasoning
        else:
            print(f"❌ [AnalysisAgent] Failed to parse response: {response}")
            return 0.5, 0.5, "Failed to parse agent response"
    except Exception as e:
        print(f"❌ [AnalysisAgent] Error parsing response: {str(e)}")
        return 0.5, 0.5, f"Error during analysis: {str(e)}"

def place_bet(probability: float):
    """ Places bet based on AI probability estimate. """
    decision = "YES" if probability > 0.5 else "NO"
    print(f"💰 [BettingAgent] Placing bet on: {decision} (Probability: {probability})")

# 🔹 Define the Trading Agent
class AutoGenAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1  

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        research_questions = generate_research_questions(market.question)
        articles = fetch_news_articles(research_questions)
        scraped_articles = [scrape_article_content(url) for url in articles if url]
        probability, confidence, reasoning = analyze_articles(market.question, scraped_articles)
        
        # Create and return a ProbabilisticAnswer object with p_yes and reasoning
        answer = ProbabilisticAnswer(
            p_yes=Probability(probability),
            confidence=Probability(confidence),
            reasoning=reasoning
        )
        
        # Log the decision
        print(f"🎯 [AutoGenAgent] Final Decision:")
        print(f"   - Probability: {probability}")
        print(f"   - Confidence: {confidence}")
        print(f"   - Reasoning: {reasoning}")
        print(f"   - Answer: {answer}")
        
        return answer

if __name__ == "__main__":
    print("🚀 Starting AI Betting Agent...")
    agent = AutoGenAgent()
    agent.run(market_type=MarketType.OMEN)

from langchain_core.prompts import ChatPromptTemplate

# Prompt to extract company name
prompt_extract_symbol = ChatPromptTemplate.from_template(
    "Extract only the company or stock name from the user's stock-related messages. "
    "Respond with the extracted company name as a single string. "
    "messages: {messages}"
)
# Prompt to answer user messages for answers without news
prompt_answer_messages_wo_news = ChatPromptTemplate.from_template(
    "You are a helpful financial expert who can provide answers to user queries.\n\n"
    "You are provided with structured stock-related data for a company: {stock_info}\n\n"
    "The user has asked: {messages}\n\n"
    "Your task:\n"
    "- If the messages can be answered using ONLY the provided data, give a clear, concise response (max 350 words).\n"
    "- If the data is insufficient to fully answer the messages, state clearly that the available information is not enough.\n\n"
    "Do not use any external knowledge or assumptions. Base your answer strictly on the given data."
)
# for messages that include news
prompt_answer_messages = ChatPromptTemplate.from_template(
    "You are a helpful financial expert who can provide answers to user queries (max 450 words).\n\n"
    "You are provided with two sources of information:\n"
    "- Structured stock-related data: {stock_info}\n"
    "- Recent news summaries (if any): {news_data}\n\n"
    "The user has asked: {messages}\n\n"
    "Your task:\n"
    "- Use the structured data as the primary source for factual, numeric, and financial information.\n"
    "- If news summaries are provided (non-empty), also consider them to enrich the answer. "
    "Clearly mention the insights derived from the news (e.g., 'According to recent news...').\n"
    "- If news_data is empty, ignore it and rely only on stock_info.\n"
    "- If neither stock_info nor news_data contain enough information to fully answer, state clearly that the available information is not enough.\n\n"
    "Constraints:\n"
    "- Keep your answer clear, concise, and under 350 words.\n"
    "- Do not use any external knowledge or assumptions beyond the provided stock_info and news_data."
)

prompt_route_all = ChatPromptTemplate.from_template(
    "You are a classifier for a stock analysis app.\n"
    "You will be given a list of messages. Consider the last message in the context of previous ones and classify it.\n\n"
    "Return only one of the following labels:\n"
    "1. 'new' — The last message is about exactly one specific stock and explicitly names it.\n"
    "2. 'follow_up' — The last message refers to a stock discussed earlier without naming it, or uses terms like 'it', 'that stock', 'the same', 'reset session'.\n"
    "3. 'other' — The last message is a greeting, unrelated to stocks, mentions more than one stock, or is about general stock market trends.\n\n"
    "Do not generate explanations or any additional text. Respond with exactly one label.\n\n"
    "Messages: {messages}"
)

prompt_handle_other = ChatPromptTemplate.from_template(
    "You are a polite assistant for a stock analysis app.\n"
    "You will be given the last user message.\n\n"
    "Rules:\n"
    "1. If it is a simple greeting (hello, hi, good morning, how are you, etc.):\n"
    "   - Greet back warmly.\n"
    "   - Mention that this app can help analyze a single stock for short-term or long-term insights.\n"
    "2. For any other message:\n"
    "   - Politely explain that this app can only help with single stock analysis and cannot answer other types of questions.\n\n"
    "Message: {message}"
)
# Prompt to determine news relevance
prompt_news_relevance = ChatPromptTemplate.from_template(
    "Decide if stock news is relevant for answering the last query in the list. "
    "Rules: "
    "- Relevant → The last query is about investment decisions, future outlook, stability, risks, or long-term prospects (e.g., 'Should I invest?', 'Is it good for long term?', 'Does the company look stable?'). "
    "- Irrelevant → The last query only asks for numeric/financial metrics (e.g., price, P/E ratio, market cap, book value), which can be answered from structured data without news. "
    "Respond with one word only: 'relevant' or 'irrelevant'. "
    "Messages: {messages} "
)
from langchain_core.prompts import ChatPromptTemplate

# Prompt to extract company name
prompt_extract_symbol = ChatPromptTemplate.from_template(
    "Extract only the company or stock name from the user's stock-related messages. "
    "Respond with the extracted company name as a single string. "
    "messages: {messages}"
)
# Prompt to answer user messages
prompt_answer_messages = ChatPromptTemplate.from_template(
    "You are a helpful financial expert who can provide answers to user queries.\n\n"
    "You are provided with structured stock-related data for a company: {stock_info}\n\n"
    "The user has asked: {messages}\n\n"
    "Your task:\n"
    "- If the messages can be answered using ONLY the provided data, give a clear, concise response (max 350 words).\n"
    "- If the data is insufficient to fully answer the messages, state clearly that the available information is not enough.\n\n"
    "Do not use any external knowledge or assumptions. Base your answer strictly on the given data."
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
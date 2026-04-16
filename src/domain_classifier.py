def detect_domain(text):

    text = text.lower()

    if any(word in text for word in ["transfer", "money", "bank", "payment"]):
        return "Fintech"

    elif any(word in text for word in ["hate", "kill", "abuse"]):
        return "Content Moderation"

    else:
        return "Chatbot"
def apply_domain_rules(domain, text, status):

    if domain == "Fintech":
        if "transfer" in text or "account" in text:
            return "🔴 HIGH RISK - Financial Action Blocked"

    elif domain == "Content Moderation":
        if status == "FAIL":
            return "🚫 Harmful Content Detected"

    elif domain == "Chatbot":
        if "ignore" in text or "reveal" in text:
            return "⚠️ Prompt Injection Attempt"

    return "✅ Safe"
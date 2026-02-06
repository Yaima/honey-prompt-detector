# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently, the following versions are being supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Honey-Prompt Detector seriously. If you have discovered a security vulnerability in this project, we appreciate your help in disclosing it to us in a responsible manner.

### Please do NOT:

- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed
- Exploit the vulnerability for malicious purposes

### Please DO:

1. **Email us directly** at yaimamvaldivia@gmail.com with:
   - A description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact of the vulnerability
   - Any suggested fixes (if you have them)

2. **Allow us time to respond** - We aim to:
   - Acknowledge receipt within 48 hours
   - Provide a more detailed response within 7 days
   - Work on a fix and keep you updated on progress

3. **Coordinate disclosure** - We will:
   - Credit you for the discovery (unless you prefer to remain anonymous)
   - Coordinate with you on the disclosure timeline
   - Notify you when the vulnerability has been fixed

## Security Best Practices

When using Honey-Prompt Detector, we recommend:

### 1. API Key Management

- **Never commit** your `.env` file or API keys to version control
- Use environment variables or secure secret management systems
- Rotate API keys regularly
- Use API keys with minimal required permissions

### 2. Dependencies

- Regularly update dependencies to get security patches
- Use `pip install -U honey-prompt-detector` to get the latest version
- Monitor security advisories for dependencies

### 3. Deployment

- Run the service in a sandboxed environment
- Use HTTPS for all API communications
- Implement rate limiting to prevent abuse
- Monitor logs for suspicious activity

### 4. Data Handling

- Do not log sensitive information
- Sanitize user inputs before processing
- Follow data retention policies
- Encrypt sensitive data at rest and in transit

### 5. LLM API Security

- Use separate API keys for production and development
- Monitor API usage for anomalies
- Implement spending limits on LLM API accounts
- Review and audit LLM API logs regularly

## Known Security Considerations

### 1. Prompt Injection Defense

While this tool is designed to detect prompt injection attacks, it is not foolproof:

- New attack vectors may emerge that are not yet detected
- Sophisticated attackers may find ways to bypass detection
- Use this as one layer in a defense-in-depth strategy

### 2. LLM API Dependencies

This project relies on third-party LLM APIs:

- API providers may have their own security vulnerabilities
- Network interception could expose sensitive data
- Always use HTTPS and validate SSL certificates

### 3. False Positives/Negatives

- The system may produce false positives (flagging benign inputs)
- The system may produce false negatives (missing real attacks)
- Regularly review and tune detection thresholds
- Implement human review for critical applications

## Security Updates

Security updates will be released as soon as possible after a vulnerability is confirmed. Updates will be announced through:

- GitHub Security Advisories
- Release notes
- Email notifications to registered users (if applicable)

## Security-Related Configuration

Important security-related configuration options:

```env
# Set appropriate confidence thresholds
CONFIDENCE_THRESHOLD=0.8
ALERT_CRITICAL_THRESHOLD=0.9

# Enable comprehensive logging for security audits
LOG_LEVEL=INFO

# Configure alert mechanisms
SLACK_WEBHOOK=your-webhook-url
EMAIL_TO=security@yourcompany.com
```

## Acknowledgments

We would like to thank the security researchers and community members who help keep Honey-Prompt Detector secure. If you have reported a security vulnerability and would like to be acknowledged, please let us know!

---

**Last Updated**: November 2025

For any questions about this security policy, please contact the maintainers.

# Security Overview

## Introduction

This document outlines the security measures implemented in the Planwise application to protect user data, prevent unauthorized access, and ensure compliance with best practices.

## Authentication and Authorization

### Authentication Methods

Planwise implements the following authentication mechanisms:

1. **JWT-based Authentication**
   - JSON Web Tokens (JWTs) are used for API authentication
   - Tokens have a configurable expiration time (default: 24 hours)
   - Refresh tokens are implemented with a longer validity period

2. **OAuth Integration**
   - Support for Google, GitHub, and other OAuth providers
   - OAuth tokens are exchanged for application-specific JWTs

3. **Password Authentication**
   - Passwords are hashed using Argon2id with appropriate work factors
   - Failed login attempts are rate-limited

### Authorization Framework

Access control is implemented using:

1. **Role-Based Access Control (RBAC)**
   - User roles include: Admin, Moderator, Regular User, Guest
   - Each role has predefined permissions

2. **Resource-Level Permissions**
   - Permissions are enforced at the API endpoint level
   - Database-level row security policies for additional protection

### Authentication Flow

```
┌─────────┐                           ┌─────────────┐                         ┌─────────────┐
│  Client │                           │    API      │                         │  Database   │
└────┬────┘                           └──────┬──────┘                         └──────┬──────┘
     │                                       │                                       │
     │ 1. Login Request                      │                                       │
     │ (username/password or OAuth)          │                                       │
     │──────────────────────────────────────>│                                       │
     │                                       │                                       │
     │                                       │ 2. Verify Credentials                 │
     │                                       │──────────────────────────────────────>│
     │                                       │                                       │
     │                                       │ 3. Return User Data                   │
     │                                       │<──────────────────────────────────────│
     │                                       │                                       │
     │ 4. Return JWT + Refresh Token         │                                       │
     │<──────────────────────────────────────│                                       │
     │                                       │                                       │
     │ 5. API Request with JWT               │                                       │
     │──────────────────────────────────────>│                                       │
     │                                       │ 6. Validate JWT                       │
     │                                       │                                       │
     │                                       │ 7. Request Data with User Context     │
     │                                       │──────────────────────────────────────>│
     │                                       │                                       │
     │                                       │ 8. Return Filtered Data               │
     │                                       │<──────────────────────────────────────│
     │ 9. Return API Response                │                                       │
     │<──────────────────────────────────────│                                       │
     │                                       │                                       │
```

## Data Protection

### Data Encryption

Planwise implements multiple layers of encryption:

1. **Data in Transit**
   - All API communications use TLS 1.3
   - HTTP Strict Transport Security (HSTS) is enforced
   - Certificate pinning is implemented in mobile applications

2. **Data at Rest**
   - Database encryption using PostgreSQL's encryption features
   - Sensitive environment variables are encrypted with AWS KMS

3. **Sensitive Data Handling**
   - PII (Personally Identifiable Information) is encrypted using AES-256
   - Encryption keys are rotated regularly

### Personal Data Minimization

We follow these principles for data minimization:

1. **Collection Limitation**
   - Only necessary data is collected from users
   - Purpose of collection is clearly stated

2. **Data Retention**
   - Automated purging of inactive user data after 24 months
   - Regular review of data retention needs

3. **Anonymization**
   - Analytics data is anonymized before processing
   - Recommendation processing uses anonymized identifiers

## API Security

### API Protection Measures

1. **Input Validation**
   - All API inputs are validated using Pydantic schemas
   - Strict type checking and constraint validation

2. **Rate Limiting**
   - IP-based rate limiting (100 requests per minute per IP)
   - User-based rate limiting (300 requests per minute per user)
   - More restrictive limits for authentication endpoints

3. **CORS Policy**
   - Restricted to known origins
   - Credentials mode properly configured

Example CORS configuration:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://planwise.io",
        "https://app.planwise.io",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600
)
```

### API Key Management

For server-to-server communication:

1. **API Key Rotation**
   - Automatic rotation every 90 days
   - Support for multiple active keys during transition

2. **Key Restrictions**
   - Keys are scoped to specific operations
   - IP address restrictions where applicable

3. **Key Storage**
   - Keys stored securely in environment variables or secret management system
   - Never committed to source code

## Infrastructure Security

### Cloud Security

1. **Network Security**
   - Production servers in private subnets
   - Public access only through load balancers
   - Security groups with minimum required ports exposed

2. **Container Security**
   - Container images scanned for vulnerabilities
   - Non-root users in containers
   - Read-only file systems where possible

3. **Serverless Function Security**
   - Least privilege IAM roles
   - Environment variable encryption
   - Function timeout and memory limits

### Infrastructure as Code Security

1. **IaC Scanning**
   - Terraform configurations scanned for security issues
   - Drift detection to prevent manual changes

2. **Secret Management**
   - AWS Secrets Manager or HashiCorp Vault for secrets
   - No secrets in infrastructure code

## Secure Development Practices

### Secure SDLC

Our development process incorporates security at every stage:

1. **Planning**
   - Threat modeling for new features
   - Security requirements definition

2. **Development**
   - Secure coding guidelines
   - Regular security training for developers

3. **Testing**
   - Static Application Security Testing (SAST)
   - Dynamic Application Security Testing (DAST)
   - Regular security reviews

4. **Deployment**
   - Secure CI/CD pipeline
   - Production approval process

### Dependency Management

1. **Dependency Scanning**
   - Automated scanning for vulnerable dependencies
   - Regular updates of dependencies

2. **Vendoring Policy**
   - Critical dependencies are vendored
   - Dependency lock files are committed

Example GitHub workflow for dependency scanning:

```yaml
name: Dependency Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sundays

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install safety
        run: pip install safety
        
      - name: Scan dependencies
        run: safety check -r requirements.txt
```

## Security Monitoring and Incident Response

### Monitoring

1. **Logging**
   - Centralized logging with Elasticsearch
   - Sensitive data is redacted from logs
   - Logs retained for 12 months

2. **Alerting**
   - Real-time alerts for suspicious activities
   - Automated responses to common attack patterns

### Incident Response

1. **Response Plan**
   - Documented incident response procedures
   - Regular tabletop exercises
   - Post-incident reviews

2. **Contact Information**
   - Security team: security@planwise.io
   - Response time goals: 
     - Critical: 2 hours
     - High: 8 hours
     - Medium: 24 hours
     - Low: 48 hours

## Compliance

### Regulatory Compliance

Planwise is designed to comply with:

1. **GDPR**
   - Data subject access rights implementation
   - Privacy by design principles
   - Data processing agreements with third parties

2. **CCPA**
   - California resident data rights
   - Do Not Sell My Personal Information implementation

### Security Standards

We follow these industry security standards:

1. **OWASP Top 10**
   - Regular audits against OWASP Top 10
   - Developer training on OWASP security risks

2. **CIS Benchmarks**
   - Server hardening following CIS guidelines
   - Regular compliance scanning

## Vulnerability Management

### Vulnerability Disclosure

1. **Responsible Disclosure Policy**
   - Public vulnerability disclosure policy
   - Bug bounty program details

2. **Reporting Process**
   - Email: security@planwise.io
   - Expected response timeline

### Vulnerability Handling

1. **Assessment**
   - CVSS scoring for all vulnerabilities
   - Priority based on impact and exploitability

2. **Remediation Timeline**
   - Critical: 24 hours
   - High: 7 days
   - Medium: 30 days
   - Low: 90 days

## Security FAQs

**Q: How are user passwords stored?**  
A: Passwords are hashed using Argon2id with salting and appropriate work factors. We never store plaintext passwords.

**Q: Does Planwise support two-factor authentication?**  
A: Yes, we support TOTP-based 2FA using authenticator apps like Google Authenticator or Authy.

**Q: How often are security audits performed?**  
A: We conduct internal security reviews quarterly and engage external auditors for penetration testing annually.

**Q: What happens in case of a data breach?**  
A: We have a documented incident response plan that includes notification procedures compliant with GDPR and other applicable regulations.

**Q: How can I report a security vulnerability?**  
A: Please email security@planwise.io with details of the vulnerability. We follow responsible disclosure practices. 
"""Helper script to create a sample PDF for testing the RAG agent."""

import pymupdf
import os

PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")


def create_pdf(filename: str, title: str, pages: list[str]):
    """Create a simple PDF with text content."""
    doc = pymupdf.open()
    for page_text in pages:
        page = doc.new_page()
        # Add title on first page
        if page_text == pages[0]:
            page.insert_text((72, 72), title, fontsize=20)
            page.insert_text((72, 110), page_text, fontsize=11)
        else:
            page.insert_text((72, 72), page_text, fontsize=11)
    filepath = os.path.join(PDF_DIR, filename)
    doc.save(filepath)
    doc.close()
    print(f"  Created: {filepath}")


# PDF 1: Company Employee Handbook
create_pdf("employee_handbook.pdf", "Acme Corp Employee Handbook 2026", [
    """Chapter 1: Leave Policy

Annual Leave: All full-time employees receive 24 days of paid annual leave per year.
Leave accrues at 2 days per month. Unused leave can be carried forward up to a maximum
of 10 days into the next calendar year. Any leave beyond 10 days will expire on March 31.

Sick Leave: Employees are entitled to 12 days of paid sick leave per year. A medical
certificate is required for absences of 3 or more consecutive days. Unused sick leave
cannot be carried forward or encashed.

Maternity Leave: Female employees are entitled to 26 weeks of paid maternity leave.
This can be taken up to 8 weeks before the expected delivery date.

Paternity Leave: Male employees are entitled to 2 weeks of paid paternity leave,
to be taken within 6 months of the child's birth.

Public Holidays: The company observes 12 public holidays per year. The list is
published in January each year. If a public holiday falls on a weekend, it is
not compensated with an additional day off unless stated otherwise.""",

    """Chapter 2: Work Hours and Remote Work

Standard Work Hours: The standard work week is 40 hours, Monday through Friday,
from 9:00 AM to 6:00 PM with a 1-hour lunch break.

Flexible Hours: Employees may request flexible working hours between 7:00 AM and
10:00 AM start time, subject to manager approval. Core hours (when everyone must
be available) are 10:00 AM to 4:00 PM.

Remote Work Policy: Employees may work remotely up to 3 days per week after
completing their 90-day probation period. Remote work requires:
- Stable internet connection (minimum 50 Mbps)
- A dedicated workspace
- Availability during core hours (10 AM - 4 PM)
- Manager approval via the HR portal

Overtime: Overtime work requires prior manager approval. Overtime is compensated
at 1.5x the regular hourly rate for weekdays and 2x for weekends and holidays.
Overtime compensation is paid in the following month's payroll.""",

    """Chapter 3: Compensation and Benefits

Salary Structure: Salaries are reviewed annually in April. The average annual
increment is 8-12% based on performance ratings.

Performance Ratings:
- Exceptional (5): Top 10% performers, eligible for 15-20% increment
- Exceeds Expectations (4): 12-15% increment
- Meets Expectations (3): 8-10% increment
- Needs Improvement (2): 0-5% increment
- Unsatisfactory (1): Performance Improvement Plan (PIP)

Health Insurance: The company provides health insurance coverage for employees
and their dependents (spouse and up to 2 children). Coverage amount:
- Employee: Rs 5,00,000
- Family floater: Rs 10,00,000

Retirement Benefits: The company contributes 12% of basic salary to the
Provident Fund (PF). Employees are also enrolled in the National Pension
Scheme (NPS) with a company contribution of 10% of basic salary.

Stock Options (ESOP): Employees at Grade L5 and above are eligible for stock
options. Vesting schedule: 25% after Year 1, then 25% each subsequent year."""
])


# PDF 2: Technical Documentation
create_pdf("api_documentation.pdf", "Acme Corp API Documentation v3.2", [
    """API Overview

Base URL: https://api.acmecorp.com/v3
Authentication: All API requests require a Bearer token in the Authorization header.
Rate Limits: Free tier: 100 requests/hour. Pro tier: 10,000 requests/hour.
Response Format: All responses are in JSON format.

Authentication Endpoint:
POST /auth/token
Body: {"email": "user@example.com", "password": "yourpassword"}
Response: {"token": "eyJhbGciOi...", "expires_in": 3600}

The token is valid for 1 hour. Use the refresh endpoint to get a new token
without re-authenticating:
POST /auth/refresh
Header: Authorization: Bearer <current_token>
Response: {"token": "newtoken...", "expires_in": 3600}""",

    """User Management Endpoints

GET /users - List all users (admin only)
Query Parameters:
  - page (int): Page number, default 1
  - limit (int): Results per page, default 20, max 100
  - role (string): Filter by role (admin, user, viewer)
Response: {"users": [...], "total": 150, "page": 1, "pages": 8}

GET /users/{id} - Get a specific user
Response: {"id": 42, "name": "Alice", "email": "alice@example.com", "role": "admin"}

POST /users - Create a new user (admin only)
Body: {"name": "Bob", "email": "bob@example.com", "role": "user"}
Response: 201 Created with user object

PUT /users/{id} - Update a user
Body: {"name": "Updated Name"} (partial updates allowed)
Response: Updated user object

DELETE /users/{id} - Delete a user (admin only)
Response: 204 No Content

Error Codes:
  400 - Bad Request (invalid parameters)
  401 - Unauthorized (missing or invalid token)
  403 - Forbidden (insufficient permissions)
  404 - Not Found
  429 - Rate Limit Exceeded""",

    """Data Endpoints

GET /data/reports - List available reports
Query Parameters:
  - start_date (string): Format YYYY-MM-DD
  - end_date (string): Format YYYY-MM-DD
  - type (string): daily, weekly, monthly

POST /data/export - Export data as CSV
Body: {"report_id": 123, "format": "csv", "filters": {"department": "engineering"}}
Response: {"download_url": "https://...", "expires_in": 300}

WebSocket Endpoint:
ws://api.acmecorp.com/v3/stream
Use for real-time data updates. Supports channels:
  - "trades" - real-time trade data
  - "alerts" - system alerts
  - "metrics" - live performance metrics

Message format: {"channel": "trades", "action": "subscribe"}
Server pushes: {"channel": "trades", "data": {...}, "timestamp": "..."}

Webhook Configuration:
POST /webhooks - Register a webhook
Body: {"url": "https://yoursite.com/hook", "events": ["user.created", "report.ready"]}
The webhook payload includes: event type, timestamp, and relevant data object."""
])

print("\nSample PDFs created! You can now run pdf_rag.py")

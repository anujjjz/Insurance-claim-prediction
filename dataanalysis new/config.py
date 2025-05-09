# Power BI Configuration
# To get these values:
# 1. Go to Azure Portal (portal.azure.com)
# 2. Navigate to Azure Active Directory > App registrations
# 3. Find your app "Insurance Claims Dashboard"
# 4. Copy the Application (client) ID
# 5. Go to Certificates & secrets
# 6. Create a new client secret and copy its value
POWERBI_CONFIG = {
    "CLIENT_ID": "YOUR_CLIENT_ID",  # Replace with the Application (client) ID from Azure Portal
    "CLIENT_SECRET": "YOUR_CLIENT_SECRET",  # Replace with the client secret value from Azure Portal
    "TENANT_ID": "d8a63e7a-515b-414d-ae44-9febcfb99c8b",
    "WORKSPACE_ID": "me",  # Using 'me' for personal workspace
    "REPORT_ID": "cde0a06b-afd0-4724-924d-e06f27378250"
} 
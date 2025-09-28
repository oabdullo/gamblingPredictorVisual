# GitHub Secrets Setup Guide

This guide explains how to securely store your NFL API key using GitHub Secrets for deployment on Streamlit Cloud.

## 🔐 What are GitHub Secrets?

GitHub Secrets are encrypted environment variables that you can use in your GitHub Actions workflows and deploy to cloud platforms like Streamlit Cloud. They keep sensitive information like API keys secure and out of your code.

## 🚀 Setting Up GitHub Secrets

### Step 1: Get Your NFL API Key

1. **Sign up for SportsData.io**:
   - Go to [sportsdata.io](https://sportsdata.io)
   - Create an account
   - Choose the NFL package
   - Get your API key

2. **Alternative APIs**:
   - [NFL.com API](https://www.nfl.com/api)
   - [ESPN API](https://developer.espn.com)
   - [Pro Football Reference](https://www.pro-football-reference.com)

### Step 2: Add Secret to GitHub Repository

1. **Go to your repository**:
   - Navigate to: https://github.com/oabdullo/gamblingPredictorVisual

2. **Access Settings**:
   - Click on "Settings" tab (top right)
   - Scroll down to "Secrets and variables" in the left sidebar
   - Click on "Actions"

3. **Add New Secret**:
   - Click "New repository secret"
   - Name: `nfl_api_key`
   - Value: Your actual API key (e.g., `abc123def456ghi789`)
   - Click "Add secret"

### Step 3: Verify Secret is Added

You should see `nfl_api_key` listed in your repository secrets. The value will be hidden for security.

## 🔧 How It Works in the Code

The application automatically checks for the API key in this order:

1. **GitHub Secrets** (when deployed on Streamlit Cloud)
2. **Environment Variables** (`NFL_API_KEY` or `SPORTSDATA_API_KEY`)
3. **Fallback to sample data** (if no API key found)

### Code Implementation

```python
def _get_api_key(self):
    """
    Get API key from GitHub secrets or environment variables.
    Priority: GitHub Secrets > Environment Variable > None
    """
    try:
        # Try to get from GitHub Secrets (when deployed on Streamlit Cloud)
        if hasattr(st, 'secrets') and 'nfl_api_key' in st.secrets:
            return st.secrets['nfl_api_key']
    except:
        pass
    
    # Try to get from environment variable
    api_key = os.getenv('NFL_API_KEY')
    if api_key:
        return api_key
    
    # Try alternative environment variable names
    api_key = os.getenv('SPORTSDATA_API_KEY')
    if api_key:
        return api_key
    
    # Return None if no API key found
    return None
```

## 🚀 Deploying with Secrets

### Streamlit Cloud Deployment

1. **Push your code to GitHub** (already done)
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Select your repository**: `oabdullo/gamblingPredictorVisual`
5. **Deploy!** - Streamlit Cloud will automatically use your GitHub secrets

### Local Development with Environment Variables

```bash
# Set environment variable
export NFL_API_KEY="your_api_key_here"

# Run the app
streamlit run app.py
```

### Docker with Environment Variables

```bash
# Run with environment variable
docker run -e NFL_API_KEY="your_api_key_here" -p 8501:8501 nfl-predictor
```

## 🔍 Verifying API Key is Working

When you run the app, you'll see one of these messages:

### ✅ API Key Found
```
✅ API Key loaded successfully - Using real NFL data!
```

### ⚠️ No API Key
```
⚠️ No API key found - Using sample data for demonstration
```

## 🛡️ Security Best Practices

1. **Never commit API keys to code**
2. **Use GitHub Secrets for cloud deployment**
3. **Use environment variables for local development**
4. **Rotate API keys regularly**
5. **Monitor API usage and costs**

## 🔧 Troubleshooting

### API Key Not Working

1. **Check the secret name**: Must be exactly `nfl_api_key`
2. **Verify the API key**: Test it with a simple curl request
3. **Check deployment logs**: Look for error messages
4. **Try environment variable**: Set `NFL_API_KEY` locally

### Testing API Key

```bash
# Test your API key
curl -H "Ocp-Apim-Subscription-Key: YOUR_API_KEY" \
     "https://api.sportsdata.io/v3/nfl/scores/json/TeamSeasonStats/2023"
```

### Common Issues

1. **Wrong secret name**: Must be `nfl_api_key` (case sensitive)
2. **API key expired**: Check with your API provider
3. **Rate limiting**: Wait and try again
4. **Network issues**: Check internet connection

## 📊 API Usage Monitoring

Most API providers offer usage dashboards where you can:
- Monitor API calls
- Check rate limits
- View billing information
- Set up alerts

## 🔄 Updating Secrets

To update your API key:

1. Go to GitHub repository settings
2. Navigate to Secrets and variables > Actions
3. Click on `nfl_api_key`
4. Click "Update"
5. Enter new API key
6. Redeploy your app

## 📝 Additional Secrets

You can add more secrets for other services:

- `database_url` - For database connections
- `email_password` - For email notifications
- `jwt_secret` - For authentication
- `stripe_key` - For payment processing

## 🎯 Next Steps

1. **Add your API key to GitHub Secrets**
2. **Deploy to Streamlit Cloud**
3. **Test with real NFL data**
4. **Monitor API usage**
5. **Set up alerts for high usage**

Your NFL prediction model will automatically use real data when the API key is available! 🏈

# NFL Game Predictor - Deployment Guide

This guide covers different ways to deploy your NFL Game Predictor application.

## 🚀 Quick Start (Local Development)

### Option 1: Direct Streamlit
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2: Using the run script
```bash
# Make script executable
chmod +x run_app.py

# Run the app
python3 run_app.py
```

## 🐳 Docker Deployment

### Local Docker
```bash
# Build the image
docker build -t nfl-predictor .

# Run the container
docker run -p 8501:8501 nfl-predictor
```

### Docker Compose
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

## ☁️ Cloud Deployment Options

### 1. Streamlit Cloud (Recommended for beginners)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/nfl-predictor.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Deploy!

### 2. Heroku

1. **Create Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   # Install Heroku CLI
   # Login to Heroku
   heroku login
   
   # Create app
   heroku create nfl-predictor-yourname
   
   # Deploy
   git push heroku main
   ```

### 3. AWS EC2

1. **Launch EC2 instance** (Ubuntu 20.04+)
2. **Install Docker**:
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo usermod -aG docker $USER
   ```

3. **Deploy**:
   ```bash
   # Clone your repo
   git clone https://github.com/yourusername/nfl-predictor.git
   cd nfl-predictor
   
   # Run with Docker Compose
   docker-compose up -d
   ```

4. **Configure Security Group**:
   - Open port 8501 for HTTP traffic
   - Access via: `http://your-ec2-ip:8501`

### 4. Google Cloud Platform

1. **Create Cloud Run service**:
   ```bash
   # Build and push to Container Registry
   gcloud builds submit --tag gcr.io/PROJECT-ID/nfl-predictor
   
   # Deploy to Cloud Run
   gcloud run deploy nfl-predictor \
     --image gcr.io/PROJECT-ID/nfl-predictor \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### 5. Azure Container Instances

1. **Build and push to Azure Container Registry**:
   ```bash
   # Login to Azure
   az login
   
   # Create resource group
   az group create --name nfl-predictor-rg --location eastus
   
   # Create container registry
   az acr create --resource-group nfl-predictor-rg --name nflpredictor --sku Basic
   
   # Build and push image
   az acr build --registry nflpredictor --image nfl-predictor .
   ```

2. **Deploy container**:
   ```bash
   az container create \
     --resource-group nfl-predictor-rg \
     --name nfl-predictor \
     --image nflpredictor.azurecr.io/nfl-predictor \
     --ports 8501 \
     --dns-name-label nfl-predictor-unique
   ```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom port
export STREAMLIT_SERVER_PORT=8501

# Optional: Set custom address
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Streamlit Configuration
The app uses `.streamlit/config.toml` for configuration:
- Theme colors
- Server settings
- Browser settings

## 📊 Monitoring and Logs

### Docker Logs
```bash
# View logs
docker-compose logs -f nfl-predictor

# View specific log lines
docker-compose logs --tail=100 nfl-predictor
```

### Health Checks
The app includes health check endpoints:
- Streamlit: `http://localhost:8501/_stcore/health`
- Docker: Built-in health check

## 🔒 Security Considerations

### Production Deployment
1. **Use HTTPS**: Configure reverse proxy (nginx) with SSL
2. **Authentication**: Add Streamlit authentication
3. **Rate Limiting**: Implement rate limiting
4. **Input Validation**: Validate all user inputs
5. **Environment Variables**: Use secrets management

### Example nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🚀 Performance Optimization

### For High Traffic
1. **Caching**: Use Redis for model caching
2. **Load Balancing**: Multiple container instances
3. **CDN**: Use CloudFlare or AWS CloudFront
4. **Database**: Store predictions in database
5. **Async Processing**: Use Celery for heavy computations

### Example with Redis caching:
```python
import redis
import pickle

# Cache model predictions
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_prediction(game_key):
    cached = redis_client.get(game_key)
    if cached:
        return pickle.loads(cached)
    return None

def cache_prediction(game_key, prediction):
    redis_client.setex(game_key, 3600, pickle.dumps(prediction))
```

## 📈 Scaling

### Horizontal Scaling
- Use Kubernetes for orchestration
- Implement load balancing
- Auto-scaling based on CPU/memory

### Vertical Scaling
- Increase container resources
- Use more powerful instances
- Optimize model performance

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy NFL Predictor

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t nfl-predictor .
    
    - name: Deploy to production
      run: |
        # Your deployment commands here
        echo "Deploying to production..."
```

## 📝 Troubleshooting

### Common Issues
1. **Port already in use**: Change port in config
2. **Memory issues**: Increase container memory
3. **Model loading slow**: Use model caching
4. **CORS errors**: Configure CORS settings

### Debug Mode
```bash
# Run with debug logging
streamlit run app.py --logger.level=debug
```

## 📞 Support

For deployment issues:
1. Check Docker logs
2. Verify port accessibility
3. Check resource usage
4. Review configuration files

## 🎯 Recommended Deployment Strategy

**For Beginners**: Streamlit Cloud
**For Production**: AWS ECS + Application Load Balancer
**For Enterprise**: Kubernetes on AWS EKS
**For Cost Optimization**: Google Cloud Run

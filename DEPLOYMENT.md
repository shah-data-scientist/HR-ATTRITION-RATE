# Deployment Guide

This guide covers deploying the HR Attrition Rate application to production environments.

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended for Production)

**Best for:** Production deployments, full control, scalability

The simplest way to deploy the entire stack with PostgreSQL database.

#### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Server with at least 2GB RAM

#### Steps

1. **Clone the repository on your server**
   ```bash
   git clone <repository-url>
   cd hr-attrition-rate
   ```

2. **Create production environment file**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` with production settings**
   ```bash
   # Production Database
   DATABASE_URL="postgresql://prod_user:secure_password@db:5432/hr_attrition_db"
   
   # API Configuration
   API_PORT="8001"
   API_HOST="0.0.0.0"
   
   # For UI to connect to API (use service name in Docker)
   API_BASE_URL="http://fastapi_app:8001"
   
   # Security
   API_TOKEN="<generate-secure-token>"
   ```

4. **Generate secure tokens**
   ```bash
   # Generate API token
   openssl rand -hex 32
   ```

5. **Build and start services**
   ```bash
   docker-compose up -d
   ```

6. **Initialize the database**
   ```bash
   docker-compose exec fastapi_app python database/init_db.py
   ```

7. **Verify deployment**
   ```bash
   # Check services are running
   docker-compose ps
   
   # Test API health
   curl http://localhost:8001/health
   
   # Test UI
   curl http://localhost:8501
   ```

#### Production Configuration

Edit `docker-compose.yml` for production:

```yaml
version: '3.8'

services:
  db:
    image: postgres:16-alpine
    restart: always
    environment:
      POSTGRES_DB: hr_attrition_db
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  fastapi_app:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    restart: always
    ports:
      - "8001:8001"
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/hr_attrition_db
      API_HOST: 0.0.0.0
      API_PORT: 8001
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit_app:
    build:
      context: .
      dockerfile: docker/Dockerfile.streamlit
    restart: always
    ports:
      - "8501:8501"
    environment:
      API_BASE_URL: http://fastapi_app:8001
    depends_on:
      - fastapi_app

volumes:
  db_data:
```

### Option 3: Cloud Deployment (AWS/Azure/GCP)

**Best for:** Enterprise production, high availability, auto-scaling

#### AWS Deployment

**Using ECS (Elastic Container Service):**

1. **Push images to ECR**
   ```bash
   # Authenticate to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and tag images
   docker build -f docker/Dockerfile.api -t hr-attrition-api .
   docker tag hr-attrition-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/hr-attrition-api:latest
   
   docker build -f docker/Dockerfile.streamlit -t hr-attrition-ui .
   docker tag hr-attrition-ui:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/hr-attrition-ui:latest
   
   # Push to ECR
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hr-attrition-api:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hr-attrition-ui:latest
   ```

2. **Set up RDS for PostgreSQL**
   - Create PostgreSQL 16 instance
   - Configure security groups
   - Note connection string

3. **Create ECS Task Definitions**
   - Define tasks for API and UI
   - Set environment variables
   - Configure resource limits

4. **Deploy to ECS**
   - Create ECS cluster
   - Deploy services
   - Configure load balancer

**Using EC2:**

1. **Launch EC2 instance**
   - Ubuntu 22.04 LTS
   - t3.medium or larger
   - Configure security groups (ports 22, 80, 443, 8001, 8501)

2. **Install Docker on EC2**
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose -y
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

3. **Deploy using Docker Compose**
   Follow Option 1 steps on the EC2 instance

4. **Set up reverse proxy (nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location /api/ {
           proxy_pass http://localhost:8001/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
       }
   }
   ```

#### Azure Deployment

**Using Azure Container Instances:**

1. **Create Azure Container Registry**
   ```bash
   az acr create --resource-group myResourceGroup --name myregistry --sku Basic
   ```

2. **Build and push images**
   ```bash
   az acr build --registry myregistry --image hr-attrition-api:latest --file docker/Dockerfile.api .
   az acr build --registry myregistry --image hr-attrition-ui:latest --file docker/Dockerfile.streamlit .
   ```

3. **Create Azure Database for PostgreSQL**
   ```bash
   az postgres flexible-server create \
       --resource-group myResourceGroup \
       --name mypostgresserver \
       --location eastus \
       --admin-user myadmin \
       --admin-password <password> \
       --version 16
   ```

4. **Deploy containers**
   ```bash
   az container create \
       --resource-group myResourceGroup \
       --name hr-attrition-api \
       --image myregistry.azurecr.io/hr-attrition-api:latest \
       --ports 8001 \
       --environment-variables DATABASE_URL=<connection-string>
   ```

### Option 4: Kubernetes Deployment

**Best for:** Large-scale production, container orchestration, microservices

For large-scale production deployments with auto-scaling and high availability.

#### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3+ (optional)

#### Deployment Steps

1. **Create namespace**
   ```bash
   kubectl create namespace hr-attrition
   ```

2. **Create secrets**
   ```bash
   kubectl create secret generic db-credentials \
       --from-literal=username=user \
       --from-literal=password=secure_password \
       -n hr-attrition
   ```

3. **Deploy PostgreSQL**
   ```yaml
   # postgres-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: postgres
     namespace: hr-attrition
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: postgres
     template:
       metadata:
         labels:
           app: postgres
       spec:
         containers:
         - name: postgres
           image: postgres:16-alpine
           env:
           - name: POSTGRES_DB
             value: hr_attrition_db
           - name: POSTGRES_USER
             valueFrom:
               secretKeyRef:
                 name: db-credentials
                 key: username
           - name: POSTGRES_PASSWORD
             valueFrom:
               secretKeyRef:
                 name: db-credentials
                 key: password
           ports:
           - containerPort: 5432
           volumeMounts:
           - name: postgres-storage
             mountPath: /var/lib/postgresql/data
         volumes:
         - name: postgres-storage
           persistentVolumeClaim:
             claimName: postgres-pvc
   ```

4. **Deploy API**
   ```yaml
   # api-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: fastapi-app
     namespace: hr-attrition
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: fastapi-app
     template:
       metadata:
         labels:
           app: fastapi-app
       spec:
         containers:
         - name: fastapi
           image: <your-registry>/hr-attrition-api:latest
           ports:
           - containerPort: 8001
           env:
           - name: DATABASE_URL
             value: postgresql://$(DB_USER):$(DB_PASS)@postgres:5432/hr_attrition_db
           - name: DB_USER
             valueFrom:
               secretKeyRef:
                 name: db-credentials
                 key: username
           - name: DB_PASS
             valueFrom:
               secretKeyRef:
                 name: db-credentials
                 key: password
   ```

5. **Deploy UI**
   ```yaml
   # ui-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: streamlit-app
     namespace: hr-attrition
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: streamlit-app
     template:
       metadata:
         labels:
           app: streamlit-app
       spec:
         containers:
         - name: streamlit
           image: <your-registry>/hr-attrition-ui:latest
           ports:
           - containerPort: 8501
           env:
           - name: API_BASE_URL
             value: http://fastapi-service:8001
   ```

6. **Create services**
   ```bash
   kubectl apply -f postgres-deployment.yaml
   kubectl apply -f api-deployment.yaml
   kubectl apply -f ui-deployment.yaml
   ```

## 🔐 Security Considerations

### SSL/TLS Configuration

**Using Let's Encrypt with Certbot:**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Environment Variables Security

- Never commit `.env` files
- Use secret management services:
  - AWS Secrets Manager
  - Azure Key Vault
  - HashiCorp Vault
- Rotate secrets regularly

### Database Security

- Use strong passwords
- Enable SSL connections
- Restrict network access
- Regular backups
- Monitor for suspicious activity

### API Security

- Enable rate limiting
- Use API authentication
- Validate all inputs
- Keep dependencies updated
- Monitor for vulnerabilities

## 📊 Monitoring

### Health Checks

**API Health:**
```bash
curl http://localhost:8001/health
```

**Database Health:**
```bash
docker-compose exec db pg_isready -U user
```

### Logging

**View logs:**
```bash
# Docker Compose
docker-compose logs -f fastapi_app
docker-compose logs -f streamlit_app

# Kubernetes
kubectl logs -f deployment/fastapi-app -n hr-attrition
kubectl logs -f deployment/streamlit-app -n hr-attrition
```

### Monitoring Tools

**Recommended tools:**
- Prometheus + Grafana for metrics
- ELK Stack for log aggregation
- Sentry for error tracking
- Datadog for APM

## 🔄 Updates and Maintenance

### Rolling Updates

**Docker Compose:**
```bash
# Pull latest code
git pull origin main

# Rebuild and restart services
docker-compose up -d --build
```

**Kubernetes:**
```bash
# Update image
kubectl set image deployment/fastapi-app fastapi=<registry>/hr-attrition-api:v2 -n hr-attrition

# Check rollout status
kubectl rollout status deployment/fastapi-app -n hr-attrition

# Rollback if needed
kubectl rollout undo deployment/fastapi-app -n hr-attrition
```

### Database Migrations

```bash
# Backup database first
docker-compose exec db pg_dump -U user hr_attrition_db > backup.sql

# Run migrations (if using Alembic)
docker-compose exec fastapi_app alembic upgrade head
```

### Backup Strategy

**Automated backups:**
```bash
#!/bin/bash
# backup-db.sh
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T db pg_dump -U user hr_attrition_db | gzip > backup_${DATE}.sql.gz

# Keep only last 7 days
find . -name "backup_*.sql.gz" -mtime +7 -delete
```

**Schedule with cron:**
```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup-db.sh
```

## 🚨 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Check resource usage
docker stats

# Restart service
docker-compose restart <service-name>
```

### Database Connection Issues

```bash
# Test connection
docker-compose exec fastapi_app python -c "from database.database import engine; print(engine.connect())"

# Check database logs
docker-compose logs db
```

### Performance Issues

- Scale replicas in Kubernetes
- Increase container resources
- Add caching layer (Redis)
- Optimize database queries
- Enable compression

## 📈 Scaling

### Horizontal Scaling

**Docker Compose:**
```bash
docker-compose up -d --scale fastapi_app=3 --scale streamlit_app=2
```

**Kubernetes:**
```bash
kubectl scale deployment fastapi-app --replicas=5 -n hr-attrition
```

### Load Balancing

Use nginx or cloud load balancers to distribute traffic across replicas.

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)

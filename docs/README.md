# Documentation Index

Complete documentation for the HR Attrition Rate project.

## Getting Started

- **[../README.md](../README.md)** - Project overview and quick start
- **[../QUICKSTART.md](../QUICKSTART.md)** - 5-minute setup guide

## Core Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and data flow
- **[../DEVELOPMENT.md](../DEVELOPMENT.md)** - Development workflow, testing, and debugging
- **[../DEPLOYMENT.md](../DEPLOYMENT.md)** - Production deployment (Docker, Cloud, Kubernetes)

## Component Documentation

- **[../api/README.md](../api/README.md)** - API endpoints and usage
- **[../scripts/README.md](../scripts/README.md)** - Utility scripts documentation

## Additional Resources

- **[REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md)** - Recent project improvements and changes
- **[archive/](archive/)** - Historical documentation (for reference only)

## Quick Links

| Topic | Document |
|-------|----------|
| First time setup | [QUICKSTART.md](../QUICKSTART.md) |
| Running locally | [README.md](../README.md#running-the-application) |
| API usage | [api/README.md](../api/README.md) |
| Docker deployment | [DEPLOYMENT.md](../DEPLOYMENT.md#docker-compose) |
| Development workflow | [DEVELOPMENT.md](../DEVELOPMENT.md) |
| System design | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Troubleshooting | [README.md](../README.md#troubleshooting) |

## Documentation Structure

```
HR-ATTRITION-RATE/
├── README.md                    # Main project documentation
├── QUICKSTART.md                # Fast setup guide
├── DEVELOPMENT.md               # Developer guide
├── DEPLOYMENT.md                # Deployment guide
├── api/
│   └── README.md               # API documentation
├── scripts/
│   └── README.md               # Scripts documentation
└── docs/
    ├── README.md               # This file
    ├── ARCHITECTURE.md         # Architecture details
    ├── REFACTOR_SUMMARY.md     # Change history
    └── archive/                # Old documentation (reference)
```

## Contributing to Documentation

When updating documentation:

1. Keep information accurate and up-to-date with code
2. Use consistent formatting and terminology
3. Include code examples where helpful
4. Update this index when adding new docs
5. Archive outdated documentation rather than deleting

## Documentation Standards

- **Port numbers**: API on 8001, UI on 8501, Database on 5432
- **Python version**: 3.12+
- **Package manager**: Poetry
- **Code examples**: Use triple backticks with language identifier
- **Links**: Use relative links within the repository

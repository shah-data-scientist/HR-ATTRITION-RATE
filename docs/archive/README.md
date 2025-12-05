# Archived Documentation

This directory contains historical documentation that has been superseded by consolidated guides but is preserved for reference.

## Purpose

These documents capture:
- Development history and decision-making process
- Detailed troubleshooting scenarios
- Specific implementation guides from earlier project phases
- Test results and bug analysis

## Current vs Archived Documentation

### Use Current Documentation (Project Root)
For active development and deployment, refer to the main documentation:

- **[README.md](../../README.md)** - Main project documentation with quick start, features, and architecture
- **[QUICKSTART.md](../../QUICKSTART.md)** - 5-minute setup guide
- **[DEVELOPMENT.md](../../DEVELOPMENT.md)** - Development workflow and best practices
- **[DEPLOYMENT.md](../../DEPLOYMENT.md)** - Production deployment guide
- **[docs/ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture details

### Archived Documents

#### Authentication & Security
- **[AUTHENTICATION_INTEGRATION.md](AUTHENTICATION_INTEGRATION.md)** - Detailed authentication setup guide
  - **Current:** See README.md "Security Setup" section and DEPLOYMENT.md
  - **Why Archived:** Information consolidated into main docs
  - **Useful For:** Understanding original authentication implementation

#### Docker Guides
- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Docker deployment profiles (local/prod/HF)
  - **Current:** See DEPLOYMENT.md "Deployment Options" section
  - **Why Archived:** Merged into unified deployment guide
  - **Useful For:** Understanding Docker profile evolution

- **[DOCKER_DEVELOPMENT_GUIDE.md](DOCKER_DEVELOPMENT_GUIDE.md)** - Comprehensive Docker development workflow
  - **Current:** See DEVELOPMENT.md "Docker Development" section
  - **Why Archived:** Key points integrated into development guide
  - **Useful For:** Detailed Docker troubleshooting and best practices

#### Testing Documentation
- **[FOUR_EMPLOYEE_TEST_RESULTS.md](FOUR_EMPLOYEE_TEST_RESULTS.md)** - Test results for 4-employee scenario
  - **Current:** See tests/TEST_README.md for current test documentation
  - **Why Archived:** Historical test results
  - **Useful For:** Understanding test evolution and baseline results

- **[MANUAL_UI_TEST_RESULTS.md](MANUAL_UI_TEST_RESULTS.md)** - Manual UI testing results
  - **Current:** Automated tests in tests/ directory
  - **Why Archived:** Now using automated testing
  - **Useful For:** Understanding manual testing approach before automation

- **[UI_TESTING_GUIDE.md](UI_TESTING_GUIDE.md)** - UI testing procedures
  - **Current:** See tests/TEST_README.md
  - **Why Archived:** Superseded by automated testing
  - **Useful For:** Original UI testing methodology

#### Bug Analysis
- **[SHAP_BUG_ANALYSIS.md](SHAP_BUG_ANALYSIS.md)** - Analysis of SHAP integration issues
  - **Current:** Issues resolved, SHAP working correctly
  - **Why Archived:** Bug fixed, kept for historical reference
  - **Useful For:** Understanding SHAP integration challenges and solutions

- **[TROUBLESHOOTING_422_ERROR.md](TROUBLESHOOTING_422_ERROR.md)** - 422 validation error troubleshooting
  - **Current:** See README.md troubleshooting section
  - **Why Archived:** Issue resolved
  - **Useful For:** Understanding validation error debugging

#### API Documentation
- **[API_SETUP_AND_USAGE.md](API_SETUP_AND_USAGE.md)** - API setup instructions
  - **Current:** See api/README.md and main README.md
  - **Why Archived:** Consolidated into main documentation
  - **Useful For:** Original API setup approach

#### Project Evolution
- **[README.old.md](README.old.md)** - Previous version of main README
  - **Current:** See README.md
  - **Why Archived:** Replaced by current README
  - **Useful For:** Tracking documentation evolution

- **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Early project solution summary
  - **Current:** See README.md and docs/ARCHITECTURE.md
  - **Why Archived:** Information integrated into current docs
  - **Useful For:** Understanding initial project design decisions

## When to Reference Archived Docs

### Good Reasons to Check Archives:
✅ Understanding historical decisions and their rationale
✅ Debugging similar issues that occurred in the past
✅ Learning from troubleshooting approaches
✅ Researching implementation evolution
✅ Writing project retrospectives or case studies

### Better Alternatives:
❌ **For deployment:** Use current DEPLOYMENT.md
❌ **For development:** Use current DEVELOPMENT.md
❌ **For API usage:** Use current api/README.md and /docs endpoint
❌ **For testing:** Use current tests/TEST_README.md
❌ **For architecture:** Use current docs/ARCHITECTURE.md

## Contributing to Archives

When archiving new documentation:
1. Add entry to this README explaining:
   - What the document covers
   - Why it was archived
   - Where to find current information
   - What historical value it provides
2. Keep original document intact (don't edit)
3. Update references in current documentation

## Document History

### 2025-12-04: Documentation Reorganization
- Moved 6 documents from root to archive
- Created consolidated guides in root
- Established archive policy

### Future Archiving
As the project evolves, additional documentation may be moved here when:
- Information is consolidated into main docs
- Issues are resolved
- Features are deprecated
- Better documentation supersedes older versions

## Related Documentation

- **[docs/REORGANIZATION_SUMMARY.md](../REORGANIZATION_SUMMARY.md)** - Details on recent project reorganization
- **[docs/REFACTOR_SUMMARY.md](../REFACTOR_SUMMARY.md)** - Development history and refactoring
- **[PROJECT_STRUCTURE.md](../../PROJECT_STRUCTURE.md)** - Current project organization

---

**Note:** These documents are preserved for reference only. Always consult current documentation for active development.

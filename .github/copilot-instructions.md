# AI Coding Agent Instructions

## Project Overview
**Course Recommendation System** - A React Native mobile app that predicts student performance and recommends courses using collaborative filtering and ML-based similarity analysis.

### Architecture
- **Frontend**: React Native (Expo) - `MainFormScreen.js` (form input) → `Results.js` (display recommendations)
- **Backend**: Node.js Express server at `recommendation-backend/` - Orchestrates Python inference
- **ML Pipeline**: Python scripts using pandas/numpy/scikit-learn for similarity calculations and predictions
- **Data**: CSV files (grades, CGPA, course correlations) + JSON for form submissions

### Data Flow
1. User submits form (subjects taken, grades, CGPA) in `MainFormScreen.js`
2. Sent via axios POST to `/submit-form` endpoint
3. Backend saves form as `data.json`, triggers Python script
4. Python computes similarity matrix, generates 6 CSV outputs (`output.csv` - `output_6.csv`)
5. Frontend fetches CSV data via `/get-csv-data/:csvId` and displays in `Results.js`

## Key Conventions

### Form Handling & Navigation
- Navigation: React Navigation stack with "Form" → "Results" flow in `App.js`
- Form state in `MainFormScreen.js` uses `useState` hooks
- Subjects list is hardcoded (~90 courses from `subjects` array)
- Pass results via route params: `navigation.navigate('Results', { results: data, preferredElective: elective })`

### Backend Communication
- Base URL is environment-based (localhost during dev, Heroku in production)
- Endpoints: `/submit-form` (POST) and `/get-csv-data/:csvId` (GET)
- CSV IDs map to: `csv1`→`output.csv` through `csv6`→`output_6.csv`
- Node backend expects Python scripts in same directory (executed via `child_process.exec()`)

### Python ML Pipeline
- Entry point: `final_modular.py` in `recommendation-backend/`
- CSV parsing uses pandas; data normalized via Jaccard similarity
- Key functions: `calculate_similarity()`, `predict_gpa()`, `get_courses_data()`
- Data files referenced: `Grade.csv`, `CGPA.csv`, `Grade_Student.csv`, `data.json` (form input)
- Output: 6 CSV files with different recommendation strategies

### UI/Styling
- Design tokens in `tokens.json` (colors, typography, spacing)
- Theme provider in `theme.js` with dark mode support
- Hidden courses in `Results.js`: "PRACTICE SCHOOL I/II", "THESIS"
- Custom components in `components/Button.js`
- Localization via `i18n/en.json`

### Build & Run
- **Frontend**: `npm start` (Expo), `npm run android`, `npm run ios`
- **Backend**: `npm start` in `recommendation-backend/` (runs on PORT 10000 by default)
- **Python**: Dependencies in `requirements.txt` (pandas, numpy, scikit-learn, torch, seaborn, matplotlib, networkx)
- Note: Backend must be running for frontend form submission to work

## Critical Points for Modifications

### Adding New Course Recommendations
- Update `subjects` array in `MainFormScreen.js`
- Ensure corresponding data in `Grade.csv`, `Grade_Student.csv` 
- Python script will auto-calculate similarity for new courses

### Debugging Form→Results Flow
- Check backend logs: form data saved to `data.json`
- Verify Python script execution completes (logs in Node server console)
- Ensure CSV files exist before frontend requests them
- Common error: CSV output files missing = Python script failed

### Cross-Backend Issues
- Two backend folders exist: `backend/` (older) and `recommendation-backend/` (active). Use `recommendation-backend/` only
- Assets folder contains `similarity_matrix.py` - may be duplicated code, verify against `recommendation-backend/similarity_matrix.py`

## Common Commands
```bash
# Frontend
cd course-reccomendation && npm start

# Backend
cd recommendation-backend && npm start && python final_modular.py

# Test backend API
curl -X POST http://localhost:10000/submit-form -H "Content-Type: application/json" -d @data.json
```

## Dependencies to Know
- **Expo**: Framework for React Native development
- **React Navigation**: Stack-based navigation (not drawer/tab)
- **Axios**: HTTP client for API calls
- **Express**: Minimal server, CORS enabled for frontend
- **Pandas**: DataFrame operations for CSV/JSON
- **Scikit-learn**: Used implicitly (check imports in Python scripts)

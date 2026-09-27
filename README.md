# PawVision AI — Cats vs Dogs

This release upgrades only the web application/API serving layer. **Keep your existing `cat_dog_model.keras` unchanged** and place it beside `main.py`.

## Features
- Premium responsive classifier UI
- Explainability Center
- AI Lab pipeline visualization
- Prediction confidence + dog probability
- Correct/Wrong feedback (because the model cannot know ground truth by itself)
- Local prediction history and CSV export
- Dark/light theme
- API health indicator

## Important confidence fix
The saved CNN has a single sigmoid output and no built-in image rescaling layer. The API now supplies the image on a 0..1 pixel scale before inference, which is the standard preprocessing expected by this type of trained image-generator pipeline. The CNN file itself is not changed.

Confidence is the model's probability for the selected class; it is **not proof that the prediction is correct**. The UI therefore lets you mark a prediction as Correct/Wrong when you know the ground truth.

## Run
Put these files together with your existing model:

```
index.html
main.py
cat_dog_model.keras   # existing model — do not replace
```

Then in the project folder:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open:

`http://127.0.0.1:8000/`

Do not double-click `index.html` while testing the API.

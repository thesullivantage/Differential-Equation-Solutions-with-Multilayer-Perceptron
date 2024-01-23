# Training Experimentation Workflow Reccomendations

1. Upload MLP_DE_collab.ipynb to Google Colab. Free-tier resources should suffice for a relatively shallow MLP model.

2. Follow cells at top for cloning & pulling my repository (or specifying your own fork)

3. Create a dev branch. Write a shell script to speed-push changes to that branch.

4. Make changes in real-time in the text editor of your choice, run shell script to push, and then run pull/module reloading cells of the Jupyter notebook to retrain model/etc.
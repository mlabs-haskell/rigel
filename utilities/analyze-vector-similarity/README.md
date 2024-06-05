# Context Vector Similarity Index Calculator

Download the bin_xx.bson.gz files into a folder and update `config.py` accordingly.

Edit the number of dumps downloaded in `main_make_db.py`
Run `just make_db`

If you want to test the DB against the original dumps, run `just test_db`

Run `just group_similarity` to find mean of similarity index between groups.

Run `just full_similarity_calculate` and then `just full_similarity_analyze` to find the pairs of entries in the descending order of similarity index.

# Course dataset provenance

The data accompany the UW-Madison COMP SCI 320 MP8 exercise, "Selling Laptops - Smart Marketing," a simulated online-retail prediction task.

The accompanying assignment README, under "Dataset," states: "We have generated 3 datasets" and identifies training, public test1, and hidden test2. The analysis requires the six supplied training and public test1 CSV files as local inputs. Raw CSV files are not bundled in this public code repository. Hidden test2 data are not available.

The local files were checked against that accompanying documentation:

| File type | Fields |
| --- | --- |
| User profiles | user_id, names, age, past_purchase_amt, badge |
| Visit logs | user_id, date, url, seconds |
| Targets | user_id, y |

Every URL value in the supplied training and public test1 logs is one of five relative exercise pages: /blender.html, /cleats.html, /keyboard.html, /laptop.html, and /tablet.html. The files do not contain email-address, phone-number, street-address, or IP-address fields. They are presented as course-generated exercise data, not records of an actual commercial campaign.

The assignment design and data source are credited to the course. This repository does not assert an additional license over those materials. Results should be interpreted within the course exercise's limitations.

To reproduce the evaluation, obtain an authorized copy of train_users.csv, train_logs.csv, train_y.csv, test1_users.csv, test1_logs.csv, and test1_y.csv. Place them beside evaluate.py or supply their directory with the --data-dir argument. The committed evaluation.json contains aggregate evaluation results from the provided course files.

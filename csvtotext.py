import csv

csv_file = "data/amazon_reviews/Reviews.csv"
txt_dir = "data/amazon_reviews_text"
primary_key_field = 'Id'

def csv2text(csv_file: str, txt_dir: str, prm_key_field: str):
    with open(csv_file, "r") as my_input_file:
            line_count = 0
            reader = csv.reader(my_input_file)
            prm_key_index = 0
            for row in reader:
                if line_count == 0:
                    prm_key_index = row.index(prm_key_field)
                    headers = ", ".join(row) +'\n'
                    print(headers)
                    line_count += 1
                else:
                    prm_key = row[prm_key_index]
                    output_path = f"{txt_dir}/{prm_key}.txt"
                    row_str = ", ".join(row)+'\n'
                    line_count += 1

                    with open(output_path, "w") as my_output_file:
                        my_output_file.write(headers + row_str)
                        my_output_file.close()

    print(f'Processed {line_count} lines.')

csv2text(csv_file, txt_dir, primary_key_field)

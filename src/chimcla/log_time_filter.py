"""
This script is used to filter log file by timestamps

"""
import os
from datetime import datetime
import argparse


def load_lines(logfile_path):
    with open(logfile_path) as fp:
        lines = fp.readlines()
    return lines

def check_date(date_string):
    result = False

    if (len(date_string) == 10):
        date_year = date_string[0:4]
        date_month = date_string[5:7]
        date_day = date_string[8:10]

        if (date_year.isdigit() and date_month.isdigit() and date_day.isdigit()):

            if (2020 <= int(date_year) <= 2030):
                if (1 <= int(date_month) <= 12):
                    if (1 <= int(date_day) <= 31):
                        result = True
    return result

def save_logfile(filter_list, begin_date, end_date, path):
    new_logfile_name = "{0}/classifier_from_{1}_to_{2}.log".format(path, begin_date, end_date)

    with open(new_logfile_name, 'w') as new_logfile:
        for line in filter_list:
            new_logfile.write(f"{line}")


class MainManager:
    def __init__(self):
        self.parse_args()
        if not (check_date(self.args.begin) and check_date(self.args.end)):
            print("Input invalid, please check the begin and end date!")
            exit()
        self.load_logfile()

    def parse_args(self):

        parser = argparse.ArgumentParser(
            prog='log_time_filter',
            description='filters the log entries within a defined time period',
        )
        parser.add_argument('--logfile', "-l", help="the logfile to be evaluated", required=True)
        parser.add_argument('--begin', "-b", help="day of the beginning (YYYY-MM-DD, 2024-08-01)", required=True)
        parser.add_argument('--end', "-e", help="day of the ending (YYYY-MM-DD, 2024-08-02)", required=True)

        self.args = parser.parse_args()

    def load_logfile(self):
        self.all_lines = load_lines(self.args.logfile)


    def main(self):

        full_list = self.all_lines #[:1000]
        filter_list = []

        begin_date = datetime.strptime(self.args.begin, "%Y-%m-%d")
        end_date = datetime.strptime(self.args.end, "%Y-%m-%d")

        for row in full_list:

            if check_date(row[:10]):
                # get date
                day_string = row[:10]

                # check if date is in range
                day_date = datetime.strptime(day_string, "%Y-%m-%d")

                if begin_date <= day_date <= end_date:
                    filter_list.append(row)
                    print(row)
                else:
                    if day_date > end_date:     #stops if there are more entries after the relevant period
                        break

        # print(*filter_list)
        save_logfile(filter_list, self.args.begin, self.args.end, os.path.dirname(self.args.logfile))

def main():
    mm = MainManager()
    mm.main()

if __name__ == "__main__":
    main()



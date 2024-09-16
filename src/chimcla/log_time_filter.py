"""
This script is used to filter log file by timestamps

"""
import os
import re
# from datetime import datetime as dtm
from datetime import datetime
import argparse


def load_lines(logfile_path):
    with open(logfile_path) as fp:
        lines = fp.readlines()
    return lines


class MainManager:
    def __init__(self):
        self.parse_args()
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
        # get lines of log file
        self.all_lines = load_lines(self.args.logfile)


    def main(self):
        print("main.")
        #print(self.all_lines[1])

        full_list = self.all_lines #[:1000]
       #print(full_list)

        filter_list = []

        for row in full_list:

            # # check if row is empty
            if row[:2] == "\n":
                print("\n")
            else:

                if row[4] == "-":
                    # get date
                    day_string = row[:10]
                    #print(day_string)        
                    
                    # check if date is in range
                    day_date = datetime.strptime(day_string, "%Y-%m-%d")
                    begin_date = datetime.strptime(self.args.begin, "%Y-%m-%d")
                    end_date = datetime.strptime(self.args.end, "%Y-%m-%d")
                    if begin_date <= day_date <= end_date:
                        filter_list.append(row)
                    else:
                        if day_date > end_date:
                            print(*filter_list, sep="\n")
                            exit()                
                            #print('no')
            
        print(*filter_list, sep="\n")

# this is executed by the cli script (see pyproject.toml)
def main():
    mm = MainManager()
    mm.main()

# obsolete but does not harm
if __name__ == "__main__":
    main()



import os


class LogFileManager:

    """
    LogFileManager

    Class to manage the log file used to track the images used to create annotation tasks so far.
    """

    def __init__(self, working_directory, restart_logging, file_name="log_queries.txt"):
        self.working_directory = working_directory
        self.log_file_name = os.path.join(self.working_directory, file_name)

        if restart_logging and os.path.exists(self.log_file_name):
            os.remove(self.log_file_name)

    def get_last_timestamp(self):
        if os.path.exists(self.log_file_name):
            with open(self.log_file_name, "r") as f:
                # read all logged timestamps
                lines = f.readlines()
                # extract last timestamp used to
                # create a task and remove '\n'
                last_timestamp = lines[-1][:-1]

        else:
            last_timestamp = "0"
            with open(self.log_file_name, "a") as f:
                f.write(last_timestamp + "\n")

        return last_timestamp

    def update_query_log(self, results):
        new_last_timestamp = int(results[-1][2])

        with open(self.log_file_name, "a") as f:
            f.write(str(new_last_timestamp) + "\n")

import sys

def error_message_details(error, error_detail:sys):
    _,_, exc_err=error_detail.exc_info()
    file_name=exc_err.tb_frame.f_code.co_filename
    line_number=exc_err.tb_lineno,
    err=str(error)

    error_message=f'Error occured in the script {file_name} line number {line_number} error message {err}'
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_datail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message, error_detail=error_datail)

    def __str__(self):
        return self.error_message
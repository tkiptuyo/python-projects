from __future__ import print_function

import os.path
import pickle

import pandas as pd
import plotly
import plotly as py

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '1XC0bLuxRnyX-_opVRW3GlfKUFnTNZ9LTdjliLahg31Q'
SAMPLE_RANGE_NAME = 'finance!A1:AX20'


def main() -> object:
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        print('Name, Major:')
        for row in values:
            # Print columns A and E, which correspond to indices 0 and 4.
            print ('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '
                  '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '
                  '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (
                      row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11],
                      row[12], row[13],
                      row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24],
                      row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34], row[35],
                      row[36], row[37], row[38], row[39], row[40], row[41], row[42], row[43], row[44], row[45], row[46],
                      row[47], row[48], row[49]))


if __name__ == '__main__':
    main()


fig = genSankey(df,cat_cols=['lvl1','lvl2','lvl3','lvl4'],value_cols='count',title='Word Etymology')
plotly.offline.plot(fig, validate=False)

from __future__ import print_function
import base64
import email
import email.policy
import decode

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/gmail.labels']
TOKEN_PATH = 'secret/token.json'
CREDENTIALS_PATH = 'secret/credentials.json'


def get_labels(credentials):
    output = {}
    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])

        if not labels:
            print('No labels found.')
            return output
        for label in labels:
            output[label['name']] = label['id']
    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')
    return output


def get_messages(credentials, logger):
    output = []
    try:
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().messages().list(userId='me').execute()
        # will require pagination
        messages = results.get('messages', [])

        if not messages:
            print('No messages found.')
            return output
        for message in messages:
            specific_logger = decode.SpecificLogger(logger=logger, name=id)
            content, labels = get_message(credentials=credentials, id=message['id'], specific_logger=specific_logger)
            output.append({
                'id': message['id'],
                'message': content,
                'labels': labels,
            })
    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')
    return output


def get_message(credentials, id, specific_logger):
    try:
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().messages().get(userId='me', id=id, format='raw').execute()
        # print(results)
        message = results.get('raw')

        if not message:
            print('No message found.')
            return None
        message = base64.urlsafe_b64decode(message)
        message = email.message_from_bytes(message, policy=email.policy.default)
        subject = message['subject']
        
        content = decode.convert_message(
            message=message,
            logger=specific_logger
        )
        subject = decode.cleanup(subject)
        content = decode.cleanup(content)
        # message = f"{subject}\n\n{content}"
        message = content  # TODO include subject
        labels = results.get('labelIds')
        return message, labels
    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')


def predictions(messages):
    import tensorflow as tf
    import dataset.dataset
    import models.export_model
    train_dir = "data/train"
    test_dir = "data/test"
    raw_train_ds, _, _ = dataset.dataset.raw_datasets(train_dir=train_dir, test_dir=test_dir)

    def get_string_labels(predicted_scores_batch):
        predicted_int_labels = tf.math.argmax(predicted_scores_batch, axis=1)
        predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
        return predicted_labels

    #
    export_model = models.export_model.load_export_model()
    predicted_scores = export_model([message['message'] for message in messages])
    predicted_labels = get_string_labels(predicted_scores)
    for message, label in zip(messages, predicted_labels):
        message['prediction'] = label.numpy()


def update_labels(credentials, ids, addLabelIds, removeLabelIds):
    try:
        body = {'ids': ids, 'addLabelIds': addLabelIds, 'removeLabelIds': removeLabelIds}
        service = build('gmail', 'v1', credentials=credentials)
        service.users().messages().batchModify(userId='me', body=body).execute()
    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')


def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())

    labels = get_labels(credentials=creds)
    label_positive, label_negative = labels['AI/AI Intressant'], labels['AI/AI Ej Intressant']

    logger = decode.Logger()
    messages = get_messages(credentials=creds, logger=logger)
    predictions(messages)
    print(messages)

    positive = [message['id'] for message in messages if message['prediction'] == b'intressanta']
    negative = [message['id'] for message in messages if message['prediction'] == b'ej_intressanta']

    print(positive)
    print(negative)
    if len(positive) > 0:
        update_labels(credentials=creds, ids=positive, addLabelIds=[label_positive], removeLabelIds=[label_negative])
    if len(negative) > 0:
        update_labels(credentials=creds, ids=negative, addLabelIds=[label_negative], removeLabelIds=[label_positive])


if __name__ == '__main__':
    main()

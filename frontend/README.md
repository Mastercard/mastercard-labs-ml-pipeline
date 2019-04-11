# web-ui

The files in this folder define a web interface that can be used to interact with a TensorFlow server

- flask_server.py
  - main server code. Handles incoming requests, and renders HTML from template
- ctp_client.py
  - code to interact with TensorFlow model server
  - takes in an image and server details, and returns the server's response
- Dockerfile
  - builds a runnable container out of the files in this directory

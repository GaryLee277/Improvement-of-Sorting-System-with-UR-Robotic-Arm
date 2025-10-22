import socket
import json
import scriptFunctions
import time
import os

googleAPIKey = "Creat your Google Cloud Vision API Key here"

image_store_folder = 'Image'
# path for ocr text storage
csv_store_folder = 'CSV'
# path for JSON file that keep diction
json_diction_path ='JSON/diction.json'
os.makedirs(image_store_folder, exist_ok=True)
os.makedirs(csv_store_folder, exist_ok=True)
os.makedirs(json_diction_path, exist_ok=True)

# camera index and resolution setting
camera_index = 1
camera_resolution_width = 3224
camera_resolution_height = 2448

# computer IP address as server
Host = "192.168.0.10"
Port = 30000

# max pairs number
part_index = 0
max_pair_number = 70
terminate_code = 99
# build a diction for parts placement and pairing
pair_diction = scriptFunctions.build_dictionary(max_pair_number)

# Input "n" to start new session
while True:
    sess = input('\nStart new session? (Enter n for new, c to continue previous, or any other key to exit)')
    if sess == 'n':
        print('Start new session')
        break
    if sess == 'c':
        print('Continue previous session')
        # load saved JSON as dictionary ([0]dictionary, [1]placed part number)
        with open(json_diction_path, 'r') as js_load:
            js_read = json.load(js_load)
        part_index = int(js_read[1]) + 1
        for js_index in range(part_index):
            pair_diction[js_index] = js_read[0][str(js_index)]
        break
    else:
        print('Exiting')
        exit(0)

# setup connection between computer and client robot using TCP/IPV4
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # server setup
    s.bind((Host, Port))
    s.listen()
    #s.settimeout(40)
    print("Awaiting robot response")

    # create object for send and receive data
    conn, client_address = s.accept()
    with conn:
        # communication between server and client
        print('Connected to robot by: ', client_address)
        try_connection = ''
        while try_connection != 'robot start':
            try_connection = bytes.decode(conn.recv(1024))
        print('Received from robot: ', try_connection)
        conn.send(str.encode('server start'))
        print('Connection setup, both server and robot initialized')

        time_count = 0

        # system start picking and placing operation
        while True:
            data_received = ''
            while data_received == '':
                data_received = bytes.decode(conn.recv(1024))
                print('Message received:', data_received)
                time.sleep(1)
                time_count += 1
                if time_count >= 30:
                    pair_diction = scriptFunctions.build_dictionary(max_pair_number)
                    time_count = 0
                    print('Waiting timeout, reset the diction.')

                print('@ Waiting state 1')
            print('Receive message from robot: ', data_received)

            if  data_received == 'robot start':
                # a new connection initialize, send start command to robot, go pick state
                conn.send(str.encode('server start'))
                print('@ Pick state...')

            if data_received == 'part not found':
                # nothing found in raw area, go waiting state
                data_received = ''
                print('@ Waiting state 2')

            if data_received == 'ocr position':
                # ocr camera capture image
                capture_image = scriptFunctions.capture_image_from_camera(camera_index, camera_resolution_width, camera_resolution_height)
                # save image for ocr and backup
                saved_image_name = scriptFunctions.save_image(capture_image, image_store_folder, part_index)
                # ocr by google vision
                ocr_text = scriptFunctions.perform_ocr_with_google_cloud(saved_image_name, googleAPIKey)
                line = scriptFunctions.extract_text_tokens(ocr_text)
                print('OCR recognized text: ', line)
                # process ocr strings
                part_number, part_keyword = scriptFunctions.check_for_six_digit_number(line)
                # calculate place position
                place_position = scriptFunctions.diction_check_fill(part_number, part_keyword, pair_diction)
                print('Place position: ', place_position)
                pair_save = [pair_diction, part_index]
                # save dictionary as JSON file
                #with open(json_diction_path, 'w+') as js_file:
                #    json.dump(pair_save, js_file)
                
                part_index = part_index + 1
                print('@ OCR state...')

                # Avoid exceed pairing space limitation
                if place_position > 70:
                    print('Exceed the maximum pair number')
                    conn.send(terminate_code.to_bytes(4, 'big'))
                else:
                    print('Place position: ', place_position)
                    conn.send(place_position.to_bytes(4, 'big'))

            if data_received == 'part placed':
                # go back to pick state
                conn.send(str.encode('go pick'))
                print('@ Place state...')
            
            time.sleep(1)

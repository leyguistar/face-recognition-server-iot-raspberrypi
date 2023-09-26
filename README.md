
# Access Control System with Artificial Intelligence for the University Network Lab

![Project Image]()

## Overview

This project is a comprehensive access control system designed for the university network lab. It incorporates cutting-edge technologies such as facial recognition, IoT, Raspberry Pi, ESP32, and Kinect. This README will provide an in-depth understanding of the project's functionality, its significance, and how it addresses a real-world problem.

## Motivation

In 2019, with the introduction of the Raspberry Pi and the potential of IoT, the motivation for this project arose. The goal was to enhance access control for the university network lab by leveraging artificial intelligence and Arduino-based solutions.

## Problem Statement

The university's network lab had a conventional access control system that relied on a physical key and a button-operated mechanism. This system became unreliable, leading to the need for a more secure and efficient solution. The key problems included:

- Limited access control options.
- The risk of getting locked inside the lab without a key.
- Lack of security for a computer lab environment.

## Key Features

- **Facial Recognition:** The system uses Kinect and AI to recognize individuals attempting to access the lab.
- **IoT Integration:** Raspberry Pi and ESP32 are used to create a networked access control system.
- **Real-time Logging:** All access attempts are logged in a database for monitoring and security purposes.

## How It Works

1. **Access Detection:** The system detects access attempts either through the physical key or via the web server.
2. **Web Server:** An ESP32 hosts a web server, allowing authorized users to control access remotely.
3. **Security Measures:** Cryptographic techniques are implemented to ensure the system's security.
4. **Database Logging:** Access events are logged in a database, creating a comprehensive record of lab access.
5. **Facial Recognition:** When a person is detected outside the lab, Kinect initiates facial recognition.
6. **Dataset Training:** A neural network is trained on a dataset of lab members' faces.
7. **Access Decision:** Based on facial recognition results, the system decides whether to grant access.

## Why It Matters

This project significantly improves the security and convenience of the university's network lab. It addresses the limitations of the previous system and demonstrates the potential of IoT and AI in access control.

## Repository Contents

- **Arduino Code:** Contains the code for the ESP32 access control unit.
- **Raspberry Pi Code:** Includes the code for image processing and facial recognition.
- **Datasets:** Contains the labeled dataset used for training the facial recognition model.
- **Documentation:** Additional documentation, wiring diagrams, and setup instructions.

## Usage

1. Clone this repository.
2. Follow the setup instructions in the respective code folders.
3. Customize the system to your specific environment and requirements.
4. Contribute to the project by improving security, efficiency, or usability.

## Acknowledgments

Special thanks to adrian from pyimagesearch.com for all the tutorials.
## License

Not yet

## Contact

For questions or collaborations, contact [Your Name] at [davida.sanchez@ciens.ucv.ve].

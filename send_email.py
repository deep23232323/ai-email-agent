import smtplib
from email.message import EmailMessage

def send_mail(receiver_email, Subject, body):
    
# 1. Define email details
    sender_email = "deep8686385@gmail.com"
    app_password = "ythk nwvi scrh zmnv"  # Replace with a secure App Password

    # 2. Create the message
    msg = EmailMessage()
    msg["Subject"] = Subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    # 3. Attempt to connect and send
    try:
        print(receiver_email, Subject, body)
        print("Connecting to the server...")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, app_password)
            server.send_message(msg)
            
        return "Email sent successfully"

    except smtplib.SMTPAuthenticationError:
        return "❌ Authentication failed. Please verify your email and App Password."

    except smtplib.SMTPConnectError:
        return "❌ Failed to connect to the SMTP server. Check your network or server address."

    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"


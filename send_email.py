import ssl, smtplib
from argparse import ArgumentParser


if __name__ == "__main__":
	parser = ArgumentParser(description="Script to send email notification")
	parser.add_argument("--message", type=str, default="test is done")
	args = parser.parse_args()

	port = 465
	context = ssl.create_default_context()
	password = "zU!T38@howU"
	
	sender = "testalert550@gmail.com"
	receiver = "ksmill327@gmail.com"
	message = f"""\
	From: {sender}\nSubject:{args.message}\n\n
	
	
	This message was automated from Python script."""
	
	with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
		server.ehlo()
		server.login(sender, password)
		server.sendmail(sender, receiver, message)	
		

from pyftpdlib import ftpserver
authorizer = ftpserver.DummyAuthorizer()
authorizer.add_user("sihirliparmakcan", "12345", "/home/can/arastir/code/tensolver/data/sihirliparmakcan", perm="elradfmw")
handler = ftpserver.FTPHandler
handler.authorizer = authorizer
address = ("127.0.0.1", 2121)
ftpd = ftpserver.FTPServer(address, handler)
ftpd.serve_forever()

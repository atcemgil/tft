import tornado.ioloop
import tornado.web
import os
import json

users=['sihirliparmakcan']
data_dir='./data/'

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Tensorver")

class DataHandler(tornado.web.RequestHandler):
#    def get(self):
#        username = self.get_argument('user')
#        found = False
#        for dirname, dirnames, filenames in os.walk('./data/'+username):
#            for d in filenames:
#                self.write(d)
#                found = True
#        if not found:
#            self.write('no files found')

    def post(self):
        username = self.get_argument('user')
        dataset = self.get_argument('dataset')
        datasetdir = './data/' + username + '/' + dataset
        if not os.path.exists(datasetdir):
            os.makedirs(datasetdir)

        datatype = self.get_argument('type')
        if datatype == 'dimension':
            dimsdir = datasetdir+'/dims'
            if not os.path.isdir(dimsdir):
                os.makedirs(dimsdir)
        
            filename = datasetdir + '/dims/' + self.get_argument('name')
            a=open(filename,'wb')
            a.write(self.get_argument('cardinality'))
            a.close()
            print "wrote " + filename

        elif datatype == 'factor':
            factorsdir = datasetdir + '/factors'
            if not os.path.isdir(factorsdir):
                os.makedirs(factorsdir)

            prop_filename = datasetdir + '/factors/' + self.get_argument('name')
            factor = {}
            factor['name'] = self.get_argument('name')
            factor['dims'] = self.get_argument('dims').split('\n')
            factor['isLatent'] = self.get_argument('isLatent')
            factor['isObserved'] = self.get_argument('isObserved')
            factor['isInput'] = self.get_argument('isInput')
            factor['isTemp'] = self.get_argument('isTemp')
            factor['isReUsed'] = self.get_argument('isReUsed')
            print "wrote " + prop_filename
            a = open(prop_filename, 'w')
            a.write(json.dumps(factor))
            a.close()

            #data_filename = datasetdir + '/factors/' + self.get_argument('name') + '.mat'
            #a = open(data_filename, 'wb')
            #a.write(self.request.files['data'][0]['body'])
            #a.close()

        self.write('ok')

class UserHandler(tornado.web.RequestHandler):
    def get(self):
        username = self.get_argument('user')
        if username in users:
            self.write('ok')
        else:
            self.write('no')

class CheckDataHandler(tornado.web.RequestHandler):
    def get(self):
        username = self.get_argument('user')
        dataset = self.get_argument('dataset')
        if username in users:
            if os.path.isdir(data_dir + username + '/' + dataset):
                self.write('ok')
            else:
                self.write('no')

class FixFtpUploadHandler(tornado.web.RequestHandler):
    def get(self):
        username = self.get_argument('user')
        dataset = self.get_argument('dataset')
        factor = self.get_argument('factor')
        uploadname = self.get_argument('uploadname')

        source_file = data_dir + username + '/' + uploadname
        if os.path.isfile(source_file):
            target_dir = data_dir + username + '/' + dataset
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            cmd = 'mv ' + source_file + ' ' + target_dir + '/factors/' + factor + '.mat'
            print cmd
            os.system(cmd)

        self.write('ok')

application = tornado.web.Application( [
        (r"/", MainHandler),
        (r"/list_data", DataHandler),
        (r"/upload_data", DataHandler),
        (r"/check_user", UserHandler),
        (r"/check_data", CheckDataHandler),
        (r"/fix_ftp_upload", FixFtpUploadHandler),
        ], debug=True)

if __name__ == "__main__":
    application.listen(8080)
    tornado.ioloop.IOLoop.instance().start()

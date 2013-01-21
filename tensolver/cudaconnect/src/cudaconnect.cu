/*
 * author: ck
 * created: 03.01.2013
 * advisor: atc
 */

#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <hiredis/hiredis.h>
#include <sys/stat.h>
#include <jsoncpp/json/reader.h>
#include "pltf.h"

const std::string data_path = "../data/";

int main(){
    redisContext *c;
    redisReply *reply;
    struct timeval timeout = { 1, 500000 }; // 1.5 seconds
    c = redisConnectWithTimeout((char*)"127.0.0.1", 6379, timeout);
    if (c->err) {
        printf("Connection error: %s\n", c->errstr);
        exit(1);
    }

    reply = (redisReply*)redisCommand(c,"BLPOP tensolver 0");
    std::string list_cmd(reply->element[1]->str, reply->element[1]->len);
    freeReplyObject(reply);

    size_t delim_pos = list_cmd.find("$_$");
    std::string username, dataset;
    username = list_cmd.substr(0, delim_pos);
    dataset = list_cmd.substr(delim_pos+3);
    //std::cout << "process dataset ." << dataset << "."
    //<< " for user " << "username ." << username << "." 
    //<< std::endl; 

    // check if folder exists
    struct stat sb;
    std::string pathname(data_path);
    pathname += username + '/' + dataset;

    if( stat(pathname.c_str() , &sb) == 0 && S_ISDIR(sb.st_mode) ){
      std::cout << "making cuda call" << std::endl;

      std::stringstream dims;
      std::string dims_filename(pathname);
      dims_filename += "/dims";
      std::ifstream df(dims_filename.c_str());
      
      dims << df.rdbuf();

      Json::Value root;   // will contains the root value after parsing.
      Json::Reader reader;
      std::cout << "string data ." << dims.str() << "." << std::endl;
      bool parsingSuccessful = reader.parse( dims.str(), root );
      if ( !parsingSuccessful ){
	// report to the user the failure and their locations in the document.
	std::cout  << "Failed to parse dimensions\n"
		   << reader.getFormattedErrorMessages();
      }else{
	const Json::Value k = root.get("dims", "UTF-8");
	for ( int index = 0; index < k.size(); ++index ){  // Iterates over the sequence elements.
	  if( index % 2 ){
	    std::cout << k[index].asInt() << std::endl;
	  }else{
	    std::cout << k[index].asString() << std::endl;
	  }
	}
      }

      
    }else{
      std::cout << "dataset " << dataset << " for user " <<  username << " not found" << std::endl;
    }
}

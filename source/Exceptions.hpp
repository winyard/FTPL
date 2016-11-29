/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef FTPL_EXCEPTIONS
#define FTPL_EXCEPTIONS

#include <exception>
#include <stdio.h>
namespace FTPL {

class FTPLException : public std::exception {
    public:
        FTPLException() {
            this->line = -1;
        };
        FTPLException(const char * message) {
            this->line = -1;
            this->message = message;
        };
        FTPLException(int line, const char * file) {
            this->line = line;
            this->file = file;
        };
        FTPLException(const char * message, int line, const char * file) {
            this->message = message;
            this->line = line;
            this->file = file;
        };
        virtual const char * what() const throw() {
            char * string = new char[255];
            if(line > -1) {
                sprintf(string, "%s \nException thrown at line %d in file %s", message, line, file);
                return string;
            } else {
                return message;
            }
        };
        void setLine(int line) {
            this->line = line;
        };
        void setFile(const char * file) {
            this->file = file;
        };
        void setMessage(const char * message) {
            this->message = message;
        };
    private:
        int line;
        const char * file;
        const char * message;
};

class IOException : public FTPLException {
    public:
        IOException() {
        };
        IOException(const char * filename, int line, const char * file) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "IO Error with file %s", filename);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };
        IOException(const char * filename) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "IO Error with file %s", filename);
            this->setMessage(message);
        };
    protected:
        const char * filename;
};

class FileNotFoundException : public IOException {
    public:
        FileNotFoundException(const char * filename) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "The following file was not found: %s", filename);
            this->setMessage(message);
        };
        FileNotFoundException(const char * filename, int line, const char * file) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "The following file was not found: %s", filename);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };
};

class OutOfBoundsException : public FTPLException {
    public:
        OutOfBoundsException(int x, int sizeX) {
            this->x = x;
            this->sizeX = sizeX;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d in image of size %d", x, sizeX);
            this->setMessage(message);
        };
        OutOfBoundsException(int x, int sizeX, int line, const char * file) {
            this->x = x;
            this->sizeX = sizeX;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d in image of size %d", x, sizeX);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };

        OutOfBoundsException(int x, int y, int sizeX, int sizeY) {
            this->x = x;
            this->y = y;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d in image of size %d, %d", x, y, sizeX, sizeY);
            this->setMessage(message);
        };
        OutOfBoundsException(int x, int y, int sizeX, int sizeY, int line, const char * file) {
            this->x = x;
            this->y = y;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d in image of size %d, %d", x, y, sizeX, sizeY);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };
        OutOfBoundsException(int x, int y, int z, int sizeX, int sizeY, int sizeZ) {
            this->x = x;
            this->y = y;
            this->y = z;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            this->sizeZ = sizeZ;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d, %d in volume of size %d, %d, %d", x, y, z, sizeX, sizeY, sizeZ);
            this->setMessage(message);
        };
        OutOfBoundsException(int x, int y, int z, int sizeX, int sizeY, int sizeZ, int line, const char * file) {
            this->x = x;
            this->y = y;
            this->y = z;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            this->sizeZ = sizeZ;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d, %d in volume of size %d, %d, %d", x, y, z, sizeX, sizeY, sizeZ);
            this->setMessage(message);
       };
    private:
        int x, y, z; // position requested
        int sizeX, sizeY, sizeZ;
};

class FTPLCompiledWithoutGTKException : public FTPLException {
    public:
        FTPLCompiledWithoutGTKException() {
            this->setMessage("FTPL was compiled without GTK and cannot complete");
        }
        FTPLCompiledWithoutGTKException(int line, const char * file) {
            this->setMessage("FTPL was compiled without GTK and cannot complete");
            this->setLine(line);
            this->setFile(file);
        }
};

class ConversionException : public FTPLException {
    public:
        ConversionException() : FTPLException() {};
        ConversionException(const char * message) : FTPLException(message) {};
        ConversionException(int line, const char * file) : FTPLException(line, file) {};
        ConversionException(const char * message, int line, const char * file) : FTPLException(message, line, file) { };
};

}; // END NAMESPACE FTPL

#endif

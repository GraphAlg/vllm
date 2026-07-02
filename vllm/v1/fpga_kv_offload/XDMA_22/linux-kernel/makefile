default:
	cd xdma  && sudo make install
	cd tools && make
	cd tests && sudo ./load_driver.sh
	sudo chmod 777 /dev/xdma*

clean:
	cd xdma  && make clean
	cd tools && make clean

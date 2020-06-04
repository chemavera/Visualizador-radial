function wait(ms){
   var start = new Date().getTime();
   var end = start;
   while(end < start + ms) {
     end = new Date().getTime();
  }
}

function clean(){
	var lines = document.getElementsByTagName("path");
	console.log(lines)
	for (var i=0;i<lines.length;i++){
		lines[i].style.pointerEvents = "none";
		console.log("Done");
	}
}
	

setInterval(clean, 1000);


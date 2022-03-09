$(document).ready(function(){
	$(document).on('click', '#upload', function(){
        var fileUpload = document.getElementById("fileUpload");
        var regex = /^([a-zA-Z0-9\s_\\.\-:])+(.csv|.txt)$/;
        if (regex.test(fileUpload.value.toLowerCase())) {
            if (typeof (FileReader) != "undefined") {
                var reader = new FileReader();
                reader.onload = function (e) {
                    var table = document.createElement("table");
					table.setAttribute('class','table table-borded table-responsive table-striped')
					table.setAttribute('id','table-list')
                    var rows = e.target.result.split("\n");
					var keys = (rows[0].split(","))
					
					
					var tblHead = document.createElement('thead');
					var rowHead = document.createElement('tr');
					table.appendChild(tblHead);
					tblHead.appendChild(rowHead)
					for(j=0; j<keys.length; j++) {
						var celHead = document.createElement('th');
						celHead.innerHTML = keys[j]
						rowHead.appendChild(celHead)
					}
					
					var tblBody = document.createElement('tbody');
					table.appendChild(tblBody);
                    for (var i =1; i < rows.length; i++) {
						
                        var cells = rows[i].split(",");
                        if (cells.length > 1) {
                            var row = tblBody.insertRow(-1);
                            for (var j = 0; j < cells.length; j++) {
                                var cell = row.insertCell(-1);
								cell.setAttribute('contenteditable', true);
                                cell.innerHTML = cells[j];
                            }
                        }
                    }
                    var dvCSV = document.getElementById("dvCSV");
                    dvCSV.innerHTML = "";
                    dvCSV.appendChild(table);
                }
                reader.readAsText(fileUpload.files[0]);
            } else {
                alert("This browser does not support HTML5.");
            }
        } else {
            alert("Please upload a valid CSV file.");
        }
    });
	
	var isClicked;
$("#train").on('click', function(){
	
	$(".loading-icon").addClass("show");

	 isClicked = $('#table-list').table2csv();
	
        $.ajax({
			headers: {
				"Access-Control-Allow-Origin": "*",
                "cache-control": "no-cache"
			   

              },
			
            type:'POST',
            url:"http://127.0.0.1:8000/",
            data: {'data': isClicked},
            success: function(result){
				$('#train').hide();  
				console.log('result',result)
				var myJsonString = JSON.stringify(result);
				
				$(".loading-icon").addClass("hide");
            $(".button").attr("disabled", false);
           
				
				
				
				var table = document.createElement("table");
				table.setAttribute('class','table table-borded table-responsive table-striped')
				table.setAttribute('id','table-out')
				
				var keys = ['item_id','artwork_medium','materials','artwork_price','artwork_year','rating']


				var tblHead = document.createElement('thead');
				var rowHead = document.createElement('tr');
				table.appendChild(tblHead);
				tblHead.appendChild(rowHead)
				for(j=0; j<keys.length; j++) {
					var celHead = document.createElement('th');
					celHead.innerHTML = keys[j]
					rowHead.appendChild(celHead)
				}

				var tblBody = document.createElement('tbody');
				table.appendChild(tblBody);
				var array =    result;
				for(var i=0; i <array.length; i++){
					var newRow = tblBody.insertRow(tblBody.length);
					for(var j=0; j<array[i].length; j++){
						var cell = newRow.insertCell(j);
						
						
						
						cell.innerHTML = array[i][j];
					}
				}	
				var outCSV = document.getElementById("outCSV");
				outCSV.innerHTML = "";
				outCSV.appendChild(table);
		
				
            },
            error: function(error){
                console.log(JSON.stringify(error));
            }
        });
});



  
	
	
	
	
	
	
	
const _trim_text = (text) => {
    return text.trim();
};
const _quote_text = (text) => {
    return '"' + text + '"';
};

function convert(tb){
    let output = "";
    let lines = [];

    $(tb).find('thead>tr').each(function () {
        let line = [];
        $(this).find('th:not(th:eq(0))').each(function () { 
            line.push(_quote_text(_trim_text($(this).text())));
        });
        lines.push(line.splice(0).toString());
    })

    $(tb).find('tbody>tr').each(function () {
        let line = [];
        $(this).find('td').each(function () {   
            if($(this).find('select').length){
                line.push(_quote_text($(this).find('option:selected').val()));
            }else if($(this).find('input').length){
                line.push(_quote_text($(this).find('input').val()));
            }
            else
            line.push(_quote_text(_trim_text($(this).text())));
        });
        lines.push(line.splice(0).toString());
    })
    output = lines.join('\n');
    return output;
};

$.fn.table2csv =  function () {
    let csv = convert(this);
	console.log(csv)
    //cases = $('#out').append($("<pre>").text(csv));   
    return csv;
};
	

	
});
	
	
	
	
	
	
	
	
	
	
	
	




$(function(){
    $('#fileForm').submit(function(e) {
        e.preventDefault(); 
        var formData = new FormData($(this)[0]); //extracting the file data 
 
        $.ajax({ //ajax sending a post request to the flask server
            url: "/",
            data: formData,
            type: 'POST',
            contentType: false,
            processData: false,
            success: function(response) { //updating the webpage with result
                $('#display').val(response.rawText || '');  
                $('#message').val(response.modText || '');  
                $('#inputForm').submit();
            },
            error: function(err) {
                console.log(err);
            }
        });
    });
});
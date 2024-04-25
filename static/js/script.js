$(document).ready(function() {
    $('#btn-predict').click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            beforeSend: function() {
                $('.loader').show();
            },
            success: function(response) {
                $('.loader').hide();
                // Check if response contains the predicted class and confidence level
                if (response && response.class && response.confidence !== undefined) {
                    var predicted_class = response.class;
                    var confidence_level = response.confidence;
                    $('#result').html('<span>Predicted Class: ' + predicted_class + '</span><br><span>Confidence Level: ' + confidence_level + '</span>');
                } else {
                    $('#result').html('Error: Invalid response from server');
                }
            },
            error: function(xhr, status, error) {
                $('.loader').hide();
                $('#result').html('Error: ' + error);
            }
        });
    });
});

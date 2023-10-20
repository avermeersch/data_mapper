function processFiles() {
    // Fetch files from the inputs
    const formData = new FormData();
    formData.append('file', document.getElementById('template-upload').files[0]);

    // Make an AJAX call to the Flask back-end
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        // Handle the response from the server here
        // For example, you might update the DOM to display the received data
    })
    .catch(error => {
        console.error('There was a problem with the AJAX call:', error);
    });
}

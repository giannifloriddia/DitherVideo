import {useState} from "react";
import * as React from "react";

function App() {

    const [selectedImage, setSelectedImage] = useState<File | null>(null);

    const handleFileChange = (event:React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedImage(file);
        }
    };

    const sendImage = async (image: File|null) => {

        if (!image) {
            alert('Please select an image first');
            return;
        }

        const formData = new FormData();
        formData.append('image', image);

        try {
            const response = await fetch('http://127.0.0.1:8000/dither_img', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.blob();
            const imageUrl = URL.createObjectURL(data);
            window.open(imageUrl, '_blank');
        } catch (error) {
            console.error('Error uploading image:', error);
        }
    };

    return (
        <>
            <h1>Dither Image</h1>
            <h2>Insert image</h2>
            <div>
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                />
            </div>
            <button onClick={() => sendImage(selectedImage)}>Submit</button>
        </>
    )
}

export default App

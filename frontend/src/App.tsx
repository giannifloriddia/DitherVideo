import {useState} from "react";
import * as React from "react";

function App() {

    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [scalingSize, setScalingSize] = useState<number>(7);

    const handleFileChange = (event:React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedImage(file);
        }
    };

    const handleScalingSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = Number(event.target.value);
        setScalingSize(value);
    };

    const sendImage = async (image: File|null, scaling_size: number) => {

        if (!image) {
            alert('Please select an image first');
            return;
        }

        const formData = new FormData();
        formData.append('image', image);
        formData.append('scaling_size', scaling_size.toString());

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
            <h1>Dither Lab</h1>
            <h2>Insert image</h2>
            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
            />
            <h3>Scaling Size:</h3>
            <input type={"number"} value={scalingSize} onChange={handleScalingSizeChange} />
            <button onClick={() => sendImage(selectedImage, scalingSize)}>Submit</button>
        </>
    )
}

export default App

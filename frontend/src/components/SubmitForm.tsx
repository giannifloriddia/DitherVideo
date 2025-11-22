import * as React from "react";
import {useState} from "react";

interface FormProps {
    onImageProcessed: (imageUrl: string) => void;
}

export default function SubmitForm({ onImageProcessed }: FormProps){

    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [scalingSize, setScalingSize] = useState<number>(7);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedImage(file);
            setError(null);
        }
    };

    const handleScalingSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setScalingSize(Number(event.target.value));
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        // 1. Prevent default form refresh
        event.preventDefault();
        setError(null);

        if (!selectedImage) {
            setError("Please select an image first.");
            return;
        }

        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('scaling_size', scalingSize.toString());

        try {
            setIsLoading(true);

            // 2. Fetch request
            const response = await fetch('http://127.0.0.1:8000/dither_img', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            // 3. Handle the blob response
            const data = await response.blob();

            // 4. Create a URL for the blob to display it in the <img> tag
            const imageUrl = URL.createObjectURL(data);

            onImageProcessed(imageUrl);

        } catch (err) {
            console.error('Error uploading image:', err);
            setError("Failed to process image. Check console for details.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
            />

            <label>
                Scaling Size: {scalingSize}
                <input
                    type="range"
                    min="1"
                    max="20"
                    value={scalingSize}
                    onChange={handleScalingSizeChange}
                    style={{ display: "block", width: "100%" }}
                />
            </label>

            {error && <p style={{ color: "red" }}>{error}</p>}

            <button type="submit" disabled={isLoading || !selectedImage}>
                {isLoading ? "Dithering..." : "Submit"}
            </button>
        </form>
    )
}
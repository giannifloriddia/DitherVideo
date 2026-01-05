import * as React from "react";
import {useState} from "react";

interface FormProps {
    onMediaProcessed: (mediaUrl: string, isVideo: boolean) => void;
}

export default function SubmitForm({ onMediaProcessed }: FormProps){

    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [scalingSize, setScalingSize] = useState<number>(7);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [fileType, setFileType] = useState<'image' | 'video' | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setError(null);
            
            if (file.type.startsWith('image/')) {
                setFileType('image');
            } else if (file.type.startsWith('video/')) {
                setFileType('video');
            } else {
                setFileType(null);
                setError("Please select an image or video file.");
            }
        }
    };

    const handleScalingSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setScalingSize(Number(event.target.value));
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        // 1. Prevent default form refresh
        event.preventDefault();
        setError(null);

        if (!selectedFile) {
            setError("Please select a file first.");
            return;
        }

        if (!fileType) {
            setError("Invalid file type. Please select an image or video.");
            return;
        }

        const formData = new FormData();
        formData.append('scaling_size', scalingSize.toString());

        // Determine endpoint and form field based on file type
        const isVideo = fileType === 'video';
        const endpoint = isVideo ? 'dither_video' : 'dither_img';
        const fieldName = isVideo ? 'video' : 'image';
        formData.append(fieldName, selectedFile);

        try {
            setIsLoading(true);

            // 2. Fetch request
            const response = await fetch(`http://127.0.0.1:8000/${endpoint}`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            // 3. Handle the blob response
            const data = await response.blob();

            // 4. Create a URL for the blob to display it
            const mediaUrl = URL.createObjectURL(data);

            onMediaProcessed(mediaUrl, isVideo);

        } catch (err) {
            console.error('Error uploading file:', err);
            setError("Failed to process file. Check console for details.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="file"
                accept="image/*,video/*"
                onChange={handleFileChange}
            />

            {fileType && (
                <p style={{ color: "#666", marginTop: "5px" }}>
                    Selected: {fileType === 'video' ? '🎬 Video' : '🖼️ Image'}
                </p>
            )}

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

            <button type="submit" disabled={isLoading || !selectedFile || !fileType}>
                {isLoading ? (fileType === 'video' ? "Dithering video (this may take a while)..." : "Dithering...") : "Submit"}
            </button>
        </form>
    )
}
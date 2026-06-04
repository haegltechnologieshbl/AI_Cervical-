/**
 * Image Compression Utility
 * Compresses images before upload to reduce transfer time and server load
 */

class ImageCompressor {
    constructor(options = {}) {
        this.maxWidth = options.maxWidth || 1920;  // Reduce resolution if needed
        this.maxHeight = options.maxHeight || 1920;
        this.quality = options.quality || 0.85;    // JPEG quality (0.1-1.0)
        this.maxSizeMB = options.maxSizeMB || 2;   // Target max file size in MB
    }

    /**
     * Compress a single image file
     */
    async compressImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (e) => {
                const img = new Image();

                img.onload = () => {
                    // Calculate new dimensions (maintain aspect ratio)
                    let width = img.width;
                    let height = img.height;

                    if (width > this.maxWidth || height > this.maxHeight) {
                        const ratio = Math.min(
                            this.maxWidth / width,
                            this.maxHeight / height
                        );
                        width = Math.round(width * ratio);
                        height = Math.round(height * ratio);
                    }

                    // Create canvas for resizing
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;

                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);

                    // Try to compress to target size
                    this.compressToSize(canvas, file.name, file.type)
                        .then(resolve)
                        .catch(reject);
                };

                img.onerror = reject;
                img.src = e.target.result;
            };

            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    /**
     * Compress canvas to target file size
     */
    async compressToSize(canvas, filename, originalType) {
        let quality = this.quality;
        let blob = null;
        let attempts = 0;
        const maxAttempts = 10;

        // Convert to JPEG for better compression (unless original is PNG with transparency)
        const outputType = (originalType === 'image/png') ? 'image/png' : 'image/jpeg';

        while (attempts < maxAttempts) {
            blob = await new Promise((resolve) => {
                canvas.toBlob(resolve, outputType, quality);
            });

            // Check if size is within target
            const sizeMB = blob.size / (1024 * 1024);

            if (sizeMB <= this.maxSizeMB) {
                break;
            }

            // Reduce quality and try again
            quality -= 0.1;
            if (quality < 0.1) {
                break;
            }
            attempts++;
        }

        // Create new File object
        const extension = outputType === 'image/png' ? 'png' : 'jpg';
        const newName = filename.replace(/\.[^/.]+$/, '') + '_compressed.' + extension;

        return new File([blob], newName, {
            type: outputType,
            lastModified: Date.now()
        });
    }

    /**
     * Compress multiple files
     */
    async compressFiles(files) {
        const results = [];

        for (let i = 0; i < files.length; i++) {
            const file = files[i];

            // Skip if already small enough
            if (file.size <= this.maxSizeMB * 1024 * 1024) {
                results.push({
                    original: file,
                    compressed: file,
                    compressed: false
                });
                continue;
            }

            try {
                const compressed = await this.compressImage(file);
                const savings = ((file.size - compressed.size) / file.size * 100).toFixed(1);

                results.push({
                    original: file,
                    compressed: compressed,
                    compressed: true,
                    savings: savings,
                    originalSize: this.formatSize(file.size),
                    compressedSize: this.formatSize(compressed.size)
                });

                // Update progress
                if (this.onProgress) {
                    this.onProgress((i + 1) / files.length, results[i]);
                }

            } catch (error) {
                console.error('Failed to compress image:', file.name, error);
                results.push({
                    original: file,
                    compressed: file,
                    compressed: false,
                    error: error.message
                });
            }
        }

        return results;
    }

    formatSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
}

// Export for use in scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImageCompressor;
}

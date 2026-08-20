import React, { useState, useRef, useCallback, useEffect } from "react";

import {
	Link2,
	Upload,
	X,
	Download,
	Wand2,
	Scissors,
	Loader2,
	AlertCircle,
	FolderOpen,
	Trash2,
	Sparkles,
} from "lucide-react";

import { removeBackground as aiRemoveBackground } from "@imgly/background-removal";

const MAX_OUTPUT_BYTES = 1024 * 1024;
const MIN_JPEG_QUALITY = 0.45;
const MAX_WORK_DIM = 2500;
const AI_MAX_DIMENSION = 2048;

const AI_CONFIG = {
	model: "isnet_fp16",
	device: "gpu",
	proxyToWorker: false,
	debug: false,
	output: {
		format: "image/png",
		quality: 1,
	},
};

const AI_FALLBACK_CONFIG = {
	model: "isnet_fp16",
	device: "cpu",
	proxyToWorker: false,
	debug: false,
	output: {
		format: "image/png",
		quality: 1,
	},
};

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);

	for (let n = 0; n < 256; n++) {
		let c = n;

		for (let k = 0; k < 8; k++) {
			c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		}

		table[n] = c >>> 0;
	}

	return table;
})();

function crc32(bytes) {
	let crc = 0xffffffff;

	for (let i = 0; i < bytes.length; i++) {
		crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
	}

	return (crc ^ 0xffffffff) >>> 0;
}

function dosDateTime(date) {
	const year = Math.max(1980, date.getFullYear());

	const dosDate =
		((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();

	const dosTime =
		(date.getHours() << 11) |
		(date.getMinutes() << 5) |
		(date.getSeconds() >> 1);

	return {
		date: dosDate & 0xffff,
		time: dosTime & 0xffff,
	};
}

function u16(n) {
	return new Uint8Array([n & 0xff, (n >>> 8) & 0xff]);
}

function u32(n) {
	return new Uint8Array([
		n & 0xff,
		(n >>> 8) & 0xff,
		(n >>> 16) & 0xff,
		(n >>> 24) & 0xff,
	]);
}

function concatBytes(chunks) {
	const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);

	const output = new Uint8Array(total);

	let offset = 0;

	for (const chunk of chunks) {
		output.set(chunk, offset);
		offset += chunk.length;
	}

	return output;
}

function strBytes(value) {
	return new TextEncoder().encode(value);
}

function buildZip(entries) {
	const { date, time } = dosDateTime(new Date());

	const localChunks = [];
	const centralChunks = [];

	let offset = 0;

	for (const entry of entries) {
		const nameBytes = strBytes(entry.name);
		const data = entry.data;

		const crc = crc32(data);
		const size = data.length;
		const flags = 0x0800;
		const compression = 0;

		const localHeader = concatBytes([
			u32(0x04034b50),
			u16(20),
			u16(flags),
			u16(compression),
			u16(time),
			u16(date),
			u32(crc),
			u32(size),
			u32(size),
			u16(nameBytes.length),
			u16(0),
			nameBytes,
		]);

		localChunks.push(concatBytes([localHeader, data]));

		const centralHeader = concatBytes([
			u32(0x02014b50),
			u16(20),
			u16(20),
			u16(flags),
			u16(compression),
			u16(time),
			u16(date),
			u32(crc),
			u32(size),
			u32(size),
			u16(nameBytes.length),
			u16(0),
			u16(0),
			u16(0),
			u16(0),
			u32(0),
			u32(offset),
			nameBytes,
		]);

		centralChunks.push(centralHeader);

		offset += localHeader.length + data.length;
	}

	const centralDirectory = concatBytes(centralChunks);

	const endOfCentralDirectory = concatBytes([
		u32(0x06054b50),
		u16(0),
		u16(0),
		u16(entries.length),
		u16(entries.length),
		u32(centralDirectory.length),
		u32(offset),
		u16(0),
	]);

	return concatBytes([...localChunks, centralDirectory, endOfCentralDirectory]);
}

function formatBytes(bytes) {
	if (bytes == null) {
		return "—";
	}

	if (bytes < 1024) {
		return `${bytes} B`;
	}

	if (bytes < 1024 * 1024) {
		return `${(bytes / 1024).toFixed(1)} KB`;
	}

	return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function stripExt(name) {
	const i = name.lastIndexOf(".");
	return i > 0 ? name.slice(0, i) : name;
}

function loadImageEl(src, crossOrigin) {
	return new Promise((resolve, reject) => {
		const img = new Image();

		if (crossOrigin) {
			img.crossOrigin = crossOrigin;
		}

		img.onload = () => resolve(img);

		img.onerror = () => {
			reject(new Error("Couldn't load image."));
		};

		img.src = src;
	});
}

function downloadBlob(blob, filename) {
	const url = URL.createObjectURL(blob);

	const a = document.createElement("a");

	a.href = url;
	a.download = filename;
	a.style.display = "none";

	document.body.appendChild(a);
	a.click();
	a.remove();

	setTimeout(() => {
		URL.revokeObjectURL(url);
	}, 4000);
}

async function selectDownloadDirectory() {
	if (
		typeof window === "undefined" ||
		typeof window.showDirectoryPicker !== "function"
	) {
		return null;
	}

	try {
		const directoryHandle = await window.showDirectoryPicker({
			mode: "readwrite",
		});

		return directoryHandle;
	} catch (error) {
		if (error?.name === "AbortError") {
			return null;
		}

		console.error("Directory selection failed:", error);

		return null;
	}
}

async function verifyDirectoryPermission(directoryHandle) {
	if (!directoryHandle) {
		return false;
	}

	try {
		if (typeof directoryHandle.queryPermission === "function") {
			const permission = await directoryHandle.queryPermission({
				mode: "readwrite",
			});

			if (permission === "granted") {
				return true;
			}
		}

		if (typeof directoryHandle.requestPermission === "function") {
			const permission = await directoryHandle.requestPermission({
				mode: "readwrite",
			});

			return permission === "granted";
		}

		return true;
	} catch {
		return false;
	}
}

async function downloadToDirectory(blob, filename, directoryHandle) {
	if (!directoryHandle) {
		throw new Error("No destination directory selected.");
	}

	const permission = await directoryHandle.queryPermission({
		mode: "readwrite",
	});

	if (permission !== "granted") {
		const requested = await directoryHandle.requestPermission({
			mode: "readwrite",
		});

		if (requested !== "granted") {
			throw new Error("Permission to write to the selected folder was denied.");
		}
	}

	const fileHandle = await directoryHandle.getFileHandle(filename, {
		create: true,
	});

	const writable = await fileHandle.createWritable();

	try {
		await writable.write(blob);
	} finally {
		await writable.close();
	}
}

function yieldToBrowser() {
	return new Promise((resolve) => {
		if (typeof requestIdleCallback === "function") {
			requestIdleCallback(() => resolve(), {
				timeout: 100,
			});
		} else {
			setTimeout(resolve, 0);
		}
	});
}

function canvasToBlob(canvas, mime, quality) {
	return new Promise((resolve, reject) => {
		try {
			canvas.toBlob(
				(blob) => {
					if (blob) {
						resolve(blob);
					} else {
						reject(new Error("Image encoding failed."));
					}
				},
				mime,
				quality,
			);
		} catch (error) {
			reject(error);
		}
	});
}

async function prepareAIInput(src) {
	const img = await loadImageEl(src);

	const originalWidth = img.naturalWidth;
	const originalHeight = img.naturalHeight;

	if (!originalWidth || !originalHeight) {
		throw new Error("Invalid image dimensions.");
	}

	const maxSide = Math.max(originalWidth, originalHeight);

	const scale = Math.min(1, AI_MAX_DIMENSION / maxSide);

	const width = Math.max(1, Math.round(originalWidth * scale));

	const height = Math.max(1, Math.round(originalHeight * scale));

	if (width === originalWidth && height === originalHeight) {
		return {
			blob: null,
			src,
			width,
			height,
			temporary: false,
		};
	}

	const canvas = document.createElement("canvas");

	canvas.width = width;
	canvas.height = height;

	const ctx = canvas.getContext("2d", {
		alpha: true,
		willReadFrequently: false,
	});

	if (!ctx) {
		throw new Error("Unable to create AI input canvas.");
	}

	ctx.imageSmoothingEnabled = true;
	ctx.imageSmoothingQuality = "high";
	ctx.clearRect(0, 0, width, height);
	ctx.drawImage(img, 0, 0, width, height);

	const blob = await canvasToBlob(canvas, "image/png", 1);

	canvas.width = 1;
	canvas.height = 1;

	const resizedUrl = URL.createObjectURL(blob);

	return {
		blob,
		src: resizedUrl,
		width,
		height,
		temporary: true,
	};
}

async function removeBackgroundAI(src) {
	const aiInput = await prepareAIInput(src);

	try {
		try {
			const result = await aiRemoveBackground(aiInput.blob || src, AI_CONFIG);

			if (!(result instanceof Blob)) {
				throw new Error("Background removal returned an invalid image.");
			}

			return result;
		} catch (primaryError) {
			console.warn("WebGPU/FP16 failed. Retrying with CPU/WASM.", primaryError);

			const fallbackResult = await aiRemoveBackground(
				aiInput.blob || src,
				AI_FALLBACK_CONFIG,
			);

			if (!(fallbackResult instanceof Blob)) {
				throw new Error("Background removal returned an invalid image.");
			}

			return fallbackResult;
		}
	} finally {
		if (aiInput.temporary && aiInput.src) {
			URL.revokeObjectURL(aiInput.src);
		}

		await yieldToBrowser();
	}
}

function drawEnhancedImage(
	img,
	width,
	height,
	{ transparent = false, enhance = false } = {},
) {
	const canvas = document.createElement("canvas");

	canvas.width = width;
	canvas.height = height;

	const ctx = canvas.getContext("2d", {
		alpha: true,
		willReadFrequently: false,
	});

	if (!ctx) {
		throw new Error("Unable to create image canvas.");
	}

	ctx.clearRect(0, 0, width, height);

	if (transparent) {
		ctx.globalCompositeOperation = "source-over";
	}

	if (enhance && !transparent) {
		ctx.filter = "contrast(1.045) saturate(1.025) brightness(1.005)";
	}

	ctx.drawImage(img, 0, 0, width, height);

	ctx.filter = "none";

	if (enhance && !transparent) {
		ctx.globalAlpha = 0.08;
		ctx.globalCompositeOperation = "source-over";
		ctx.filter = "contrast(1.08)";

		ctx.drawImage(img, 0, 0, width, height);

		ctx.globalAlpha = 1;
		ctx.filter = "none";
	}

	return canvas;
}

async function encodeJpegUnderLimit(img, width, height, quality, enhance) {
	let currentWidth = width;
	let currentHeight = height;

	let currentQuality = Math.min(1, Math.max(MIN_JPEG_QUALITY, quality));

	let bestBlob = null;
	let bestWidth = width;
	let bestHeight = height;

	for (let attempt = 0; attempt < 7; attempt++) {
		const canvas = drawEnhancedImage(img, currentWidth, currentHeight, {
			enhance,
			transparent: false,
		});

		const blob = await canvasToBlob(canvas, "image/jpeg", currentQuality);

		canvas.width = 1;
		canvas.height = 1;

		if (!bestBlob || blob.size < bestBlob.size) {
			bestBlob = blob;
			bestWidth = currentWidth;
			bestHeight = currentHeight;
		}

		if (blob.size <= MAX_OUTPUT_BYTES) {
			return {
				blob,
				width: currentWidth,
				height: currentHeight,
				mime: "image/jpeg",
			};
		}

		await yieldToBrowser();

		if (currentQuality > MIN_JPEG_QUALITY) {
			currentQuality = Math.max(MIN_JPEG_QUALITY, currentQuality - 0.1);

			continue;
		}

		currentWidth = Math.max(320, Math.round(currentWidth * 0.82));

		currentHeight = Math.max(320, Math.round(currentHeight * 0.82));

		currentQuality = Math.min(0.82, quality);
	}

	return {
		blob: bestBlob,
		width: bestWidth,
		height: bestHeight,
		mime: "image/jpeg",
	};
}

async function encodePngUnderLimit(img, width, height, enhance) {
	let currentWidth = width;
	let currentHeight = height;

	let bestBlob = null;
	let bestWidth = width;
	let bestHeight = height;

	for (let attempt = 0; attempt < 6; attempt++) {
		const canvas = drawEnhancedImage(img, currentWidth, currentHeight, {
			enhance: false,
			transparent: true,
		});

		const blob = await canvasToBlob(canvas, "image/png", 1);

		canvas.width = 1;
		canvas.height = 1;

		if (!bestBlob || blob.size < bestBlob.size) {
			bestBlob = blob;
			bestWidth = currentWidth;
			bestHeight = currentHeight;
		}

		if (blob.size <= MAX_OUTPUT_BYTES) {
			return {
				blob,
				width: currentWidth,
				height: currentHeight,
				mime: "image/png",
			};
		}

		await yieldToBrowser();

		const scale = attempt === 0 ? 0.88 : 0.78;

		currentWidth = Math.max(320, Math.round(currentWidth * scale));

		currentHeight = Math.max(320, Math.round(currentHeight * scale));
	}

	return {
		blob: bestBlob,
		width: bestWidth,
		height: bestHeight,
		mime: "image/png",
	};
}

async function processImage(item, settings) {
	const img = await loadImageEl(
		item.workingSrc,
		item.needsCrossOrigin ? "anonymous" : undefined,
	);

	let w = img.naturalWidth;
	let h = img.naturalHeight;

	if (!w || !h) {
		throw new Error("Invalid image dimensions.");
	}

	const cap = settings.optimize
		? Math.min(settings.maxDimension, MAX_WORK_DIM)
		: MAX_WORK_DIM;

	const scale = Math.min(1, cap / Math.max(w, h));

	w = Math.max(1, Math.round(w * scale));

	h = Math.max(1, Math.round(h * scale));

	if (settings.removeBg) {
		try {
			const aiBlob = await removeBackgroundAI(item.workingSrc);

			if (!(aiBlob instanceof Blob)) {
				throw new Error("Background removal returned an invalid image.");
			}

			const aiUrl = URL.createObjectURL(aiBlob);

			try {
				const transparentImg = await loadImageEl(aiUrl);

				if (!settings.optimize && !settings.enhance) {
					return {
						blob: aiBlob,
						mime: "image/png",
						width: transparentImg.naturalWidth,
						height: transparentImg.naturalHeight,
					};
				}

				let outputWidth = transparentImg.naturalWidth;

				let outputHeight = transparentImg.naturalHeight;

				if (settings.optimize) {
					const outputCap = Math.min(settings.maxDimension, MAX_WORK_DIM);

					const outputScale = Math.min(
						1,
						outputCap / Math.max(outputWidth, outputHeight),
					);

					outputWidth = Math.max(1, Math.round(outputWidth * outputScale));

					outputHeight = Math.max(1, Math.round(outputHeight * outputScale));
				}

				return await encodePngUnderLimit(
					transparentImg,
					outputWidth,
					outputHeight,
					false,
				);
			} finally {
				URL.revokeObjectURL(aiUrl);

				await yieldToBrowser();
			}
		} catch (error) {
			console.error("AI background removal failed:", error);

			throw new Error(
				error?.message ||
					"AI background removal failed. Please try the image again.",
			);
		}
	}

	if (settings.optimize || settings.enhance) {
		return await encodeJpegUnderLimit(
			img,
			w,
			h,
			settings.quality,
			settings.enhance,
		);
	}

	const canvas = document.createElement("canvas");

	canvas.width = w;
	canvas.height = h;

	const ctx = canvas.getContext("2d");

	if (!ctx) {
		throw new Error("Unable to create image canvas.");
	}

	ctx.drawImage(img, 0, 0, w, h);

	const mime = "image/jpeg";

	const blob = await canvasToBlob(canvas, mime, settings.quality);

	canvas.width = 1;
	canvas.height = 1;

	return {
		blob,
		mime,
		width: w,
		height: h,
	};
}

let nextId = 1;

export default function ImageGrabberOptimizer() {
	const [items, setItems] = useState([]);
	const [isDragging, setIsDragging] = useState(false);
	const [urlInput, setUrlInput] = useState("");
	const [processing, setProcessing] = useState(false);
	const [processingIndex, setProcessingIndex] = useState(0);
	const [processingTotal, setProcessingTotal] = useState(0);
	const [removeBg, setRemoveBg] = useState(false);
	const [optimize, setOptimize] = useState(true);
	const [enhance, setEnhance] = useState(true);
	const [quality, setQuality] = useState(0.82);
	const [maxDimension, setMaxDimension] = useState(1600);
  const [downloadDirectory, setDownloadDirectory] = useState(null);
	const [selectingDirectory, setSelectingDirectory] = useState(false);
	const [isSelectingDirectory, setIsSelectingDirectory] = useState(false);

	const fileInputRef = useRef(null);

	const supportsDirectoryDownload =
		typeof window !== "undefined" &&
		typeof window.showDirectoryPicker === "function";

	useEffect(() => {
		return () => {
			items.forEach((it) => {
				if (it.objectUrl) {
					URL.revokeObjectURL(it.objectUrl);
				}

				if (it.processed?.url) {
					URL.revokeObjectURL(it.processed.url);
				}
			});
		};
	}, []);

  const selectDownloadDirectory = useCallback(async () => {
		if (
			typeof window === "undefined" ||
			typeof window.showDirectoryPicker !== "function"
		) {
			return null;
		}

		try {
			setSelectingDirectory(true);

			const directoryHandle = await window.showDirectoryPicker({
				mode: "readwrite",
				startIn: "downloads",
			});

			const permission = await directoryHandle.queryPermission({
				mode: "readwrite",
			});

			if (permission !== "granted") {
				const requestedPermission = await directoryHandle.requestPermission({
					mode: "readwrite",
				});

				if (requestedPermission !== "granted") {
					throw new Error(
						"Permission to write to the selected folder was denied.",
					);
				}
			}

			setDownloadDirectory(directoryHandle);

			return directoryHandle;
		} catch (error) {
			if (error?.name === "AbortError") {
				return null;
			}

			console.error("Folder selection failed:", error);

			alert(error?.message || "Unable to select the download folder.");

			return null;
		} finally {
			setSelectingDirectory(false);
		}
	}, []);

	const addFiles = useCallback((fileList) => {
		const files = Array.from(fileList).filter((f) =>
			f.type.startsWith("image/"),
		);

		if (!files.length) {
			return;
		}

		setItems((prev) => [
			...prev,
			...files.map((file) => {
				const objectUrl = URL.createObjectURL(file);

				return {
					id: nextId++,
					name: file.name,
					originalSize: file.size,
					objectUrl,
					workingSrc: objectUrl,
					needsCrossOrigin: false,
					loadStatus: "ready",
					loadError: null,
					processStatus: "idle",
					processError: null,
					processed: null,
				};
			}),
		]);
	}, []);

	const addUrl = useCallback(async (rawUrl) => {
		const url = rawUrl.trim();

		if (!url) {
			return;
		}

		let name = "image";

		try {
			const u = new URL(url);

			name = decodeURIComponent(
				u.pathname.split("/").filter(Boolean).pop() || "image",
			);
		} catch {}

		const id = nextId++;

		setItems((prev) => [
			...prev,
			{
				id,
				name,
				originalSize: null,
				objectUrl: null,
				workingSrc: url,
				needsCrossOrigin: false,
				loadStatus: "loading",
				loadError: null,
				processStatus: "idle",
				processError: null,
				processed: null,
			},
		]);

		try {
			const res = await fetch(url, {
				mode: "cors",
			});

			if (!res.ok) {
				throw new Error("Request failed");
			}

			const blob = await res.blob();

			if (!blob.type.startsWith("image/")) {
				throw new Error("Not an image");
			}

			const objectUrl = URL.createObjectURL(blob);

			setItems((prev) =>
				prev.map((it) =>
					it.id === id
						? {
								...it,
								objectUrl,
								workingSrc: objectUrl,
								needsCrossOrigin: false,
								originalSize: blob.size,
								loadStatus: "ready",
							}
						: it,
				),
			);
		} catch {
			setItems((prev) =>
				prev.map((it) =>
					it.id === id
						? {
								...it,
								workingSrc: url,
								needsCrossOrigin: true,
								loadStatus: "ready",
								loadError: "cors-risk",
							}
						: it,
				),
			);
		}
	}, []);

	const removeItem = useCallback(
		(id) => {
			if (processing) {
				return;
			}

			setItems((prev) => {
				const it = prev.find((x) => x.id === id);

				if (it?.objectUrl) {
					URL.revokeObjectURL(it.objectUrl);
				}

				if (it?.processed?.url) {
					URL.revokeObjectURL(it.processed.url);
				}

				return prev.filter((x) => x.id !== id);
			});
		},
		[processing],
	);

	const clearAll = useCallback(() => {
		if (processing) {
			return;
		}

		items.forEach((it) => {
			if (it.objectUrl) {
				URL.revokeObjectURL(it.objectUrl);
			}

			if (it.processed?.url) {
				URL.revokeObjectURL(it.processed.url);
			}
		});

		setItems([]);
	}, [items, processing]);

	const handleDrop = useCallback(
		(e) => {
			e.preventDefault();

			setIsDragging(false);

			if (e.dataTransfer.files && e.dataTransfer.files.length) {
				addFiles(e.dataTransfer.files);

				return;
			}

			const uri =
				e.dataTransfer.getData("text/uri-list") ||
				e.dataTransfer.getData("text/plain");

			if (uri && /^https?:\/\//i.test(uri.trim())) {
				addUrl(uri);
			}
		},
		[addFiles, addUrl],
	);

	const handleProcess = useCallback(async () => {
		if (!removeBg && !optimize && !enhance) {
			return;
		}

		if (processing) {
			return;
		}

		const current = [...items];

		if (!current.length) {
			return;
		}

		setProcessing(true);
		setProcessingIndex(0);
		setProcessingTotal(current.length);

		const settings = {
			removeBg,
			optimize,
			enhance,
			quality,
			maxDimension,
		};

		setItems((prev) =>
			prev.map((it) => ({
				...it,
				processStatus: "queued",
				processError: null,
			})),
		);

		try {
			for (let index = 0; index < current.length; index++) {
				const item = current[index];

				setProcessingIndex(index + 1);

				setItems((prev) =>
					prev.map((it) =>
						it.id === item.id
							? {
									...it,
									processStatus: "processing",
								}
							: it,
					),
				);

				await yieldToBrowser();

				try {
					const result = await processImage(item, settings);

					const url = URL.createObjectURL(result.blob);

					setItems((prev) =>
						prev.map((it) =>
							it.id === item.id
								? {
										...it,
										processStatus: "done",
										processed: {
											blob: result.blob,
											url,
											size: result.blob.size,
											width: result.width,
											height: result.height,
											mime: result.mime,
										},
									}
								: it,
						),
					);
				} catch (err) {
					console.error(`Failed processing ${item.name}:`, err);

					setItems((prev) =>
						prev.map((it) =>
							it.id === item.id
								? {
										...it,
										processStatus: "error",
										processError: err?.message || "Processing failed.",
									}
								: it,
						),
					);
				}

				await yieldToBrowser();

				await new Promise((resolve) => setTimeout(resolve, 50));
			}
		} finally {
			setProcessing(false);

			await yieldToBrowser();
		}
	}, [items, processing, removeBg, optimize, enhance, quality, maxDimension]);

	const handleSelectDownloadDirectory = useCallback(async () => {
		if (!supportsDirectoryDownload || isSelectingDirectory) {
			return;
		}

		setIsSelectingDirectory(true);

		try {
			const directory = await selectDownloadDirectory();

			if (directory) {
				setDownloadDirectory(directory);
			}
		} finally {
			setIsSelectingDirectory(false);
		}
	}, [supportsDirectoryDownload, isSelectingDirectory]);

const handleDownload = useCallback(async () => {
	const done = items.filter(
		(it) =>
			it.processStatus === "done" &&
			it.processed &&
			it.processed.blob instanceof Blob &&
			it.processed.blob.size > 0,
	);

	if (!done.length) {
		return;
	}

	const supportsDirectoryPicker =
		typeof window !== "undefined" &&
		typeof window.showDirectoryPicker === "function" &&
		window.isSecureContext;

	let directoryHandle = downloadDirectory;

	try {
		if (supportsDirectoryPicker) {
			if (!directoryHandle) {
				directoryHandle = await selectDownloadDirectory();

				if (!directoryHandle) {
					return;
				}
			}
		}

		if (done.length === 1) {
			const it = done[0];

			const ext = it.processed.mime === "image/png" ? "png" : "jpg";

			const filename = `${stripExt(it.name)}-optimized.${ext}`;

			if (supportsDirectoryPicker && directoryHandle) {
				await downloadToDirectory(it.processed.blob, filename, directoryHandle);
			} else {
				downloadBlob(it.processed.blob, filename);
			}

			return;
		}

		const usedNames = new Set();
		const entries = [];

		for (const it of done) {
			const ext = it.processed.mime === "image/png" ? "png" : "jpg";

			const baseName = stripExt(it.name);

			let filename = `${baseName}-optimized.${ext}`;

			let counter = 2;

			while (usedNames.has(filename)) {
				filename = `${baseName}-optimized (${counter}).${ext}`;

				counter++;
			}

			usedNames.add(filename);

			const arrayBuffer = await it.processed.blob.arrayBuffer();

			const data = new Uint8Array(arrayBuffer);

			if (!data.length) {
				continue;
			}

			entries.push({
				name: filename,
				data,
			});

			await yieldToBrowser();
		}

		if (!entries.length) {
			throw new Error("No valid processed images were available.");
		}

		const zipBytes = buildZip(entries);

		if (
			zipBytes.length < 22 ||
			zipBytes[0] !== 0x50 ||
			zipBytes[1] !== 0x4b ||
			zipBytes[2] !== 0x03 ||
			zipBytes[3] !== 0x04
		) {
			throw new Error("Generated ZIP has an invalid signature.");
		}

		const zipBlob = new Blob([zipBytes.buffer], {
			type: "application/zip",
		});

		if (supportsDirectoryPicker && directoryHandle) {
			await downloadToDirectory(
				zipBlob,
				"optimized-images.zip",
				directoryHandle,
			);
		} else {
			downloadBlob(zipBlob, "optimized-images.zip");
		}
	} catch (error) {
		console.error("Download failed:", error);

		if (error?.name === "AbortError") {
			return;
		}

		if (error?.name === "NotAllowedError") {
			setDownloadDirectory(null);

			alert("Permission to write to the selected folder was denied.");

			return;
		}

		alert(error?.message || "Unable to save the processed files.");
	}
}, [items, downloadDirectory, selectDownloadDirectory]);

	const doneItems = items.filter(
		(it) => it.processStatus === "done" && it.processed,
	);

	const errorItems = items.filter((it) => it.processStatus === "error");

	const totalOriginal = doneItems.reduce(
		(s, it) => s + (it.originalSize || 0),
		0,
	);

	const totalProcessed = doneItems.reduce(
		(s, it) => s + (it.processed?.size || 0),
		0,
	);

	const savedPct =
		totalOriginal > 0
			? Math.max(0, Math.round((1 - totalProcessed / totalOriginal) * 100))
			: null;

	const overLimitCount = doneItems.filter(
		(it) => it.processed.size > MAX_OUTPUT_BYTES,
	).length;

	const colors = {
		bg: "#14181B",
		panel: "#1B2023",
		panelAlt: "#20262A",
		border: "#2B3237",
		text: "#ECEDE9",
		textDim: "#8B9298",
		accent: "#E8A33D",
		accentDim: "#6B5330",
		good: "#6FCF97",
		bad: "#E8615A",
	};

	return (
		<div
			style={{
				background: colors.bg,
				color: colors.text,
				minHeight: "100%",
			}}
			className="w-full rounded-xl p-6 font-sans">
			<style>{`
				.checker {
					background-image:
						linear-gradient(45deg, #333 25%, transparent 25%),
						linear-gradient(-45deg, #333 25%, transparent 25%),
						linear-gradient(45deg, transparent 75%, #333 75%),
						linear-gradient(-45deg, transparent 75%, #333 75%);
					background-size: 12px 12px;
					background-position:
						0 0,
						0 6px,
						6px -6px,
						-6px 0;
					background-color: #1a1a1a;
				}

				.mono {
					font-variant-numeric: tabular-nums;
					font-family:
						ui-monospace,
						SFMono-Regular,
						Menlo,
						Consolas,
						monospace;
				}
			`}</style>

			<div className="mb-6">
				<div
					className="text-xs tracking-widest uppercase mb-1"
					style={{
						color: colors.accent,
					}}>
					Image Lab
				</div>

				<h1 className="text-2xl font-bold tracking-tight">
					Grabber, background remover &amp; optimizer
				</h1>

				<p
					className="text-sm mt-1"
					style={{
						color: colors.textDim,
					}}>
					Drop an image link or browse local files, then enhance, remove the
					background and/or compress before downloading.
				</p>
			</div>

			<div
				onDragOver={(e) => {
					e.preventDefault();
					setIsDragging(true);
				}}
				onDragLeave={() => setIsDragging(false)}
				onDrop={handleDrop}
				style={{
					border: `1.5px dashed ${isDragging ? colors.accent : colors.border}`,
					background: isDragging ? colors.accentDim + "33" : colors.panel,
					borderRadius: 12,
				}}
				className="p-8 flex flex-col items-center justify-center text-center transition-colors">
				<Upload
					size={28}
					style={{
						color: colors.textDim,
					}}
					className="mb-3"
				/>

				<p className="text-sm mb-1">Drag image files or an image link here</p>

				<p
					className="text-xs mb-4"
					style={{
						color: colors.textDim,
					}}>
					or
				</p>

				<button
					onClick={() => fileInputRef.current?.click()}
					disabled={processing}
					style={{
						background: colors.panelAlt,
						border: `1px solid ${colors.border}`,
						color: colors.text,
					}}
					className="px-4 py-2 rounded-lg text-sm flex items-center gap-2 hover:brightness-110 disabled:opacity-40">
					<FolderOpen size={16} />
					Browse local files
				</button>

				<input
					ref={fileInputRef}
					type="file"
					accept="image/*"
					multiple
					hidden
					onChange={(e) => {
						if (e.target.files) {
							addFiles(e.target.files);
						}

						e.target.value = "";
					}}
				/>

				<div className="flex items-center gap-2 mt-5 w-full max-w-md">
					<Link2
						size={16}
						style={{
							color: colors.textDim,
						}}
					/>

					<input
						value={urlInput}
						disabled={processing}
						onChange={(e) => setUrlInput(e.target.value)}
						onKeyDown={(e) => {
							if (e.key === "Enter" && urlInput.trim()) {
								addUrl(urlInput);

								setUrlInput("");
							}
						}}
						placeholder="https://example.com/photo.jpg"
						style={{
							background: colors.panelAlt,
							border: `1px solid ${colors.border}`,
							color: colors.text,
						}}
						className="flex-1 rounded-lg px-3 py-2 text-sm outline-none disabled:opacity-40"
					/>

					<button
						disabled={processing}
						onClick={() => {
							if (urlInput.trim()) {
								addUrl(urlInput);

								setUrlInput("");
							}
						}}
						style={{
							background: colors.panelAlt,
							border: `1px solid ${colors.border}`,
							color: colors.text,
						}}
						className="px-3 py-2 rounded-lg text-sm hover:brightness-110 disabled:opacity-40">
						Add
					</button>
				</div>
			</div>

			{items.length > 0 && (
				<div className="mt-6">
					<div className="flex items-center justify-between mb-2">
						<span
							className="text-xs uppercase tracking-widest"
							style={{
								color: colors.textDim,
							}}>
							{items.length} image
							{items.length > 1 ? "s" : ""} loaded
						</span>

						<button
							onClick={clearAll}
							disabled={processing}
							style={{
								color: colors.textDim,
							}}
							className="text-xs flex items-center gap-1 hover:text-white disabled:opacity-40">
							<Trash2 size={12} />
							Clear all
						</button>
					</div>

					<div
						className="grid gap-3"
						style={{
							gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
						}}>
						{items.map((it) => (
							<div
								key={it.id}
								style={{
									background: colors.panel,
									border: `1px solid ${colors.border}`,
									borderRadius: 10,
								}}
								className="p-2 relative">
								<button
									onClick={() => removeItem(it.id)}
									disabled={processing}
									style={{
										background: colors.panelAlt,
										color: colors.textDim,
									}}
									className="absolute top-1 right-1 rounded-full p-1 hover:text-white z-10 disabled:opacity-40"
									aria-label={`Remove ${it.name}`}>
									<X size={12} />
								</button>

								<div
									className="checker rounded-md overflow-hidden mb-2"
									style={{
										aspectRatio: "1/1",
									}}>
									<img
										src={it.processed?.url || it.workingSrc}
										alt={it.name}
										className="w-full h-full object-contain"
									/>
								</div>

								<p className="text-xs truncate mb-0.5" title={it.name}>
									{it.name}
								</p>

								<p
									className="text-[11px] mono"
									style={{
										color: colors.textDim,
									}}>
									{formatBytes(it.originalSize)}

									{it.processed && (
										<>
											{" → "}
											<span
												style={{
													color: colors.good,
												}}>
												{formatBytes(it.processed.size)}
											</span>
										</>
									)}
								</p>

								{it.processStatus === "queued" && (
									<div
										className="flex items-center gap-1 mt-1 text-[11px]"
										style={{
											color: colors.textDim,
										}}>
										Waiting…
									</div>
								)}

								{it.processStatus === "processing" && (
									<div
										className="flex items-center gap-1 mt-1 text-[11px]"
										style={{
											color: colors.accent,
										}}>
										<Loader2 size={11} className="animate-spin" />
										Processing
									</div>
								)}

								{it.processStatus === "done" && (
									<div
										className="flex items-center gap-1 mt-1 text-[11px]"
										style={{
											color: colors.good,
										}}>
										✓ Complete
									</div>
								)}

								{it.processStatus === "error" && (
									<div
										className="flex items-start gap-1 mt-1 text-[11px]"
										style={{
											color: colors.bad,
										}}>
										<AlertCircle size={11} className="mt-0.5 shrink-0" />

										<span>{it.processError}</span>
									</div>
								)}

								{it.loadError === "cors-risk" &&
									it.processStatus === "idle" && (
										<div
											className="flex items-start gap-1 mt-1 text-[11px]"
											style={{
												color: colors.textDim,
											}}>
											<AlertCircle size={11} className="mt-0.5 shrink-0" />

											<span>
												May fail to process — source blocks cross-origin access.
											</span>
										</div>
									)}
							</div>
						))}
					</div>
				</div>
			)}

			<div
				style={{
					background: colors.panel,
					border: `1px solid ${colors.border}`,
					borderRadius: 12,
				}}
				className="mt-6 p-4">
				<div className="flex flex-wrap gap-5">
					<label className="flex items-center gap-2 text-sm cursor-pointer">
						<input
							type="checkbox"
							checked={removeBg}
							disabled={processing}
							onChange={(e) => setRemoveBg(e.target.checked)}
						/>
						<Scissors size={14} />
						Remove background
					</label>

					<label className="flex items-center gap-2 text-sm cursor-pointer">
						<input
							type="checkbox"
							checked={optimize}
							disabled={processing}
							onChange={(e) => setOptimize(e.target.checked)}
						/>
						<Wand2 size={14} />
						Optimize &amp; compress
					</label>

					<label className="flex items-center gap-2 text-sm cursor-pointer">
						<input
							type="checkbox"
							checked={enhance}
							disabled={processing}
							onChange={(e) => setEnhance(e.target.checked)}
						/>
						<Sparkles size={14} />
						Enhance image quality
					</label>
				</div>

				{removeBg && (
					<p
						className="text-xs mt-2"
						style={{
							color: colors.textDim,
						}}>
						AI Processing may take several seconds per image. Works best on
						images with a clear subject and background.
					</p>
				)}

				{enhance && (
					<p
						className="text-xs mt-2"
						style={{
							color: colors.textDim,
						}}>
						Applies subtle clarity, contrast and sharpening while keeping normal
						image output close to the 1 MB target.
					</p>
				)}

				<div className="grid sm:grid-cols-2 gap-4 mt-4">
					<div>
						<div
							className="flex justify-between text-xs mb-1"
							style={{
								color: colors.textDim,
							}}>
							<span>Max dimension</span>

							<span className="mono">
								{maxDimension}
								px
							</span>
						</div>

						<select
							disabled={!optimize || processing}
							value={maxDimension}
							onChange={(e) => setMaxDimension(Number(e.target.value))}
							style={{
								background: colors.panelAlt,
								border: `1px solid ${colors.border}`,
								color: colors.text,
							}}
							className="w-full rounded-lg px-2 py-1.5 text-sm disabled:opacity-40">
							<option value={800}>800px</option>

							<option value={1280}>1280px</option>

							<option value={1600}>1600px</option>

							<option value={2000}>2000px</option>

							<option value={2500}>2500px (original size if smaller)</option>
						</select>
					</div>

					<div>
						<div
							className="flex justify-between text-xs mb-1"
							style={{
								color: colors.textDim,
							}}>
							<span>JPEG quality</span>

							<span className="mono">{Math.round(quality * 100)}%</span>
						</div>

						<input
							type="range"
							min={0.45}
							max={1}
							step={0.05}
							disabled={!optimize || removeBg || processing}
							value={quality}
							onChange={(e) => setQuality(Number(e.target.value))}
							className="w-full disabled:opacity-40"
						/>

						{removeBg && (
							<p
								className="text-[11px] mt-1"
								style={{
									color: colors.textDim,
								}}>
								Doesn't apply — transparent images export as PNG.
							</p>
						)}
					</div>
				</div>

				<div
					className="mt-4 rounded-lg px-3 py-2 text-xs flex items-center gap-2"
					style={{
						background: colors.panelAlt,
						color: colors.textDim,
					}}>
					<Sparkles size={13} />
					Output target:
					<strong
						style={{
							color: colors.good,
						}}>
						≤ 1 MB
					</strong>
					per image
				</div>
			</div>

			<div className="mt-6 flex flex-wrap items-center gap-3">
				<button
					onClick={handleProcess}
					disabled={
						!items.length || processing || (!removeBg && !optimize && !enhance)
					}
					style={{
						background: colors.accent,
						color: "#211505",
					}}
					className="px-5 py-2.5 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-40">
					{processing ? (
						<>
							<Loader2 size={16} className="animate-spin" />
							Processing {processingIndex}/{processingTotal}
						</>
					) : (
						<>
							<Wand2 size={16} />
							Process images
						</>
					)}
				</button>

				{supportsDirectoryDownload && (
					<button
						onClick={handleSelectDownloadDirectory}
						disabled={processing || isSelectingDirectory}
						style={{
							background: colors.panelAlt,
							border: `1px solid ${colors.border}`,
							color: colors.text,
						}}
						className="px-5 py-2.5 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-40">
						{isSelectingDirectory ? (
							<Loader2 size={16} className="animate-spin" />
						) : (
							<FolderOpen size={16} />
						)}

						{downloadDirectory
							? "Download Folder Selected"
							: "Select Download Folder"}
					</button>
				)}

				<button
					onClick={handleDownload}
					disabled={!doneItems.length || processing || selectingDirectory}
					style={{
						background: colors.panelAlt,
						border: `1px solid ${colors.border}`,
						color: colors.text,
					}}
					className="px-5 py-2.5 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-40">
					{selectingDirectory ? (
						<>
							<Loader2 size={16} className="animate-spin" />
							Select folder...
						</>
					) : (
						<>
							<FolderOpen size={16} />

							{doneItems.length > 1
								? `Choose folder & download ${doneItems.length} images`
								: "Choose folder & download"}
						</>
					)}
				</button>

				{processing && (
					<span
						className="text-xs"
						style={{
							color: colors.textDim,
						}}>
						Processing one image at a time to keep your browser responsive.
					</span>
				)}

				{!removeBg && !optimize && !enhance && (
					<span
						className="text-xs"
						style={{
							color: colors.textDim,
						}}>
						Choose at least one action above.
					</span>
				)}
			</div>

			{downloadDirectory && (
				<div
					className="mt-2 text-xs flex items-center gap-1"
					style={{
						color: colors.good,
					}}>
					<FolderOpen size={12} />
					Files will be saved automatically to the selected folder.
				</div>
			)}

			{!supportsDirectoryDownload && (
				<div
					className="mt-2 text-xs"
					style={{
						color: colors.textDim,
					}}>
					Your browser does not support direct folder downloads. Files will use
					your browser's normal download location.
				</div>
			)}

			{errorItems.length > 0 && (
				<p
					className="text-xs mt-2"
					style={{
						color: colors.bad,
					}}>
					{errorItems.length} of {items.length} image
					{items.length > 1 ? "s" : ""} failed to process and won't be included
					in the download.
				</p>
			)}

			{doneItems.length > 0 && (
				<div
					style={{
						background: colors.panel,
						border: `1px solid ${colors.border}`,
						borderRadius: 12,
					}}
					className="mt-6 p-4 flex flex-wrap gap-6">
					<div>
						<div
							className="text-[11px] uppercase tracking-widest"
							style={{
								color: colors.textDim,
							}}>
							Original
						</div>

						<div className="mono text-lg">{formatBytes(totalOriginal)}</div>
					</div>

					<div>
						<div
							className="text-[11px] uppercase tracking-widest"
							style={{
								color: colors.textDim,
							}}>
							Processed
						</div>

						<div
							className="mono text-lg"
							style={{
								color: colors.good,
							}}>
							{formatBytes(totalProcessed)}
						</div>
					</div>

					{savedPct !== null && (
						<div>
							<div
								className="text-[11px] uppercase tracking-widest"
								style={{
									color: colors.textDim,
								}}>
								Saved
							</div>

							<div
								className="mono text-lg"
								style={{
									color: colors.accent,
								}}>
								{savedPct}%
							</div>
						</div>
					)}

					<div>
						<div
							className="text-[11px] uppercase tracking-widest"
							style={{
								color: colors.textDim,
							}}>
							Size limit
						</div>

						<div
							className="mono text-lg"
							style={{
								color: overLimitCount > 0 ? colors.bad : colors.good,
							}}>
							{overLimitCount > 0 ? `${overLimitCount} over` : "≤ 1 MB"}
						</div>
					</div>
				</div>
			)}
		</div>
	);
}

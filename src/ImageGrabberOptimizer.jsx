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

/* ---------------------------------------------------------------------- */
/* CONSTANTS                                                              */
/* ---------------------------------------------------------------------- */

const MAX_OUTPUT_BYTES = 1024 * 1024;
const MIN_JPEG_QUALITY = 0.45;

/*
 * Maximum dimensions for normal image processing.
 */
const MAX_WORK_DIM = 2500;

/*
 * Maximum dimension sent to the segmentation model.
 *
 * 2048 gives substantially better edge detail than very small inference
 * sizes while preventing huge images from consuming excessive memory.
 */
const AI_MAX_DIMENSION = 2048;

/*
 * Primary model.
 *
 * isnet_fp16 provides a better quality/speed balance than the quantized
 * quint8 model.
 */
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

/*
 * CPU/WASM fallback.
 *
 * This is intentionally a separate configuration because some browsers
 * can report GPU support but still fail during WebGPU/FP16 initialization.
 */
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

/* ---------------------------------------------------------------------- */
/* ZIP HELPERS                                                            */
/* ---------------------------------------------------------------------- */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);

  for (let n = 0; n < 256; n++) {
    let c = n;

    for (let k = 0; k < 8; k++) {
      c = (c & 1)
        ? 0xedb88320 ^ (c >>> 1)
        : c >>> 1;
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

function writeU16(value) {
  const out = new Uint8Array(2);
  const view = new DataView(out.buffer);

  view.setUint16(0, value >>> 0, true);

  return out;
}

function writeU32(value) {
  const out = new Uint8Array(4);
  const view = new DataView(out.buffer);

  view.setUint32(0, value >>> 0, true);

  return out;
}

function concatBytes(chunks) {
  const total = chunks.reduce(
    (sum, chunk) => sum + chunk.length,
    0
  );

  const result = new Uint8Array(total);

  let offset = 0;

  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  return result;
}

function strBytes(str) {
  return new TextEncoder().encode(str);
}

function dosDateTime(date) {
  const year = Math.max(
    1980,
    Math.min(2107, date.getFullYear())
  );

  const dosDate =
    ((year - 1980) << 9) |
    ((date.getMonth() + 1) << 5) |
    date.getDate();

  const dosTime =
    (date.getHours() << 11) |
    (date.getMinutes() << 5) |
    Math.floor(date.getSeconds() / 2);

  return {
    date: dosDate & 0xffff,
    time: dosTime & 0xffff,
  };
}

/**
 * Creates a standards-compliant ZIP archive.
 *
 * Important:
 * - Uses STORE compression (method 0)
 * - Uses UTF-8 filename flag
 * - Writes valid local file headers
 * - Writes valid central directory headers
 * - Writes valid EOCD record
 * - Does not use data descriptors
 * - Does not use ZIP64
 */
function buildZip(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error("Cannot create an empty ZIP archive.");
  }

  if (entries.length > 0xffff) {
    throw new Error("Too many files for a standard ZIP archive.");
  }

  const now = new Date();
  const { date, time } = dosDateTime(now);

  const localParts = [];
  const centralParts = [];

  let localOffset = 0;

  for (const entry of entries) {
    if (!entry || typeof entry.name !== "string") {
      throw new Error("Invalid ZIP entry name.");
    }

    if (!(entry.data instanceof Uint8Array)) {
      throw new Error(`Invalid ZIP data for "${entry.name}".`);
    }

    const nameBytes = strBytes(entry.name);
    const data = entry.data;

    if (nameBytes.length > 0xffff) {
      throw new Error(`ZIP filename is too long: ${entry.name}`);
    }

    if (data.length > 0xffffffff) {
      throw new Error(`File is too large for standard ZIP: ${entry.name}`);
    }

    if (localOffset > 0xffffffff) {
      throw new Error("ZIP archive is too large for standard ZIP.");
    }

    const crc = crc32(data);

    /*
     * General purpose bit flag:
     *
     * Bit 11 = UTF-8 filename
     */
    const flags = 0x0800;

    /*
     * Compression method:
     *
     * 0 = STORE / no compression
     */
    const compressionMethod = 0;

    /* --------------------------------------------------------------- */
    /* LOCAL FILE HEADER                                                */
    /* --------------------------------------------------------------- */

    const localHeader = concatBytes([
      writeU32(0x04034b50), // Local file header signature
      writeU16(20),         // Version needed to extract
      writeU16(flags),      // General purpose bit flag
      writeU16(compressionMethod),
      writeU16(time),
      writeU16(date),
      writeU32(crc),
      writeU32(data.length),
      writeU32(data.length),
      writeU16(nameBytes.length),
      writeU16(0),         // Extra field length
      nameBytes,
    ]);

    localParts.push(localHeader);
    localParts.push(data);

    /* --------------------------------------------------------------- */
    /* CENTRAL DIRECTORY HEADER                                         */
    /* --------------------------------------------------------------- */

    const centralHeader = concatBytes([
      writeU32(0x02014b50), // Central directory signature

      writeU16(20),         // Version made by
      writeU16(20),         // Version needed to extract

      writeU16(flags),
      writeU16(compressionMethod),

      writeU16(time),
      writeU16(date),

      writeU32(crc),

      writeU32(data.length),
      writeU32(data.length),

      writeU16(nameBytes.length),
      writeU16(0),          // Extra field length
      writeU16(0),          // Comment length

      writeU16(0),          // Disk number start
      writeU16(0),          // Internal attributes
      writeU32(0),          // External attributes

      writeU32(localOffset),

      nameBytes,
    ]);

    centralParts.push(centralHeader);

    localOffset += localHeader.length + data.length;
  }

  const localDirectory = concatBytes(localParts);
  const centralDirectory = concatBytes(centralParts);

  /* --------------------------------------------------------------- */
  /* END OF CENTRAL DIRECTORY                                         */
  /* --------------------------------------------------------------- */

  const endOfCentralDirectory = concatBytes([
    writeU32(0x06054b50),

    writeU16(0), // Disk number
    writeU16(0), // Disk containing central directory

    writeU16(entries.length),
    writeU16(entries.length),

    writeU32(centralDirectory.length),
    writeU32(localDirectory.length),

    writeU16(0), // ZIP comment length
  ]);

  return concatBytes([
    localDirectory,
    centralDirectory,
    endOfCentralDirectory,
  ]);
}

/* ---------------------------------------------------------------------- */
/* GENERAL HELPERS                                                        */
/* ---------------------------------------------------------------------- */

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

	document.body.appendChild(a);
	a.click();
	a.remove();

	setTimeout(() => {
		URL.revokeObjectURL(url);
	}, 4000);
}

/* ---------------------------------------------------------------------- */
/* BROWSER YIELD                                                          */
/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */
/* CANVAS -> BLOB                                                         */
/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */
/* PREPARE AI INPUT                                                       */
/* ---------------------------------------------------------------------- */

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

	/*
	 * Image already small enough.
	 *
	 * Keep the original source so we don't introduce an unnecessary
	 * resize before segmentation.
	 */
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

	/*
	 * White background is deliberately NOT added here.
	 * The model should receive the original pixels.
	 */
	ctx.clearRect(0, 0, width, height);

	ctx.drawImage(img, 0, 0, width, height);

	/*
	 * PNG is used for the resized AI input.
	 *
	 * JPEG compression can destroy fine hair/fur/object edges before
	 * segmentation, which directly hurts mask quality.
	 */
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

/* ---------------------------------------------------------------------- */
/* BACKGROUND REMOVAL                                                     */
/* ---------------------------------------------------------------------- */

async function removeBackgroundAI(src) {
	const aiInput = await prepareAIInput(src);

	try {
		console.log("[BG] Starting background removal:", {
			model: AI_CONFIG.model,
			device: AI_CONFIG.device,
			width: aiInput.width,
			height: aiInput.height,
		});

		/*
		 * First attempt:
		 *
		 * WebGPU + FP16.
		 */
		try {
			const result = await aiRemoveBackground(aiInput.blob || src, AI_CONFIG);

			if (!(result instanceof Blob)) {
				throw new Error("Background removal returned an invalid image.");
			}

			return result;
		} catch (primaryError) {
			console.warn(
				"[BG] WebGPU/FP16 failed. Retrying with CPU/WASM.",
				primaryError,
			);

			/*
			 * Important:
			 *
			 * We do NOT switch to quint8 here because the goal is better
			 * edge quality. quint8 can introduce segmentation artifacts.
			 */
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

/* ---------------------------------------------------------------------- */
/* DRAW ENHANCED IMAGE                                                    */
/* ---------------------------------------------------------------------- */

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

	/*
	 * Only apply enhancement to normal image output.
	 *
	 * For transparent AI cutouts, aggressive enhancement can alter
	 * semi-transparent edge pixels and create halos.
	 */
	if (enhance && !transparent) {
		ctx.filter = "contrast(1.045) saturate(1.025) brightness(1.005)";
	}

	ctx.drawImage(img, 0, 0, width, height);

	ctx.filter = "none";

	/*
	 * Very mild sharpening for normal images.
	 */
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

/* ---------------------------------------------------------------------- */
/* JPEG <= 1 MB                                                           */
/* ---------------------------------------------------------------------- */

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

		/*
		 * Reduce JPEG quality first.
		 */
		if (currentQuality > MIN_JPEG_QUALITY) {
			currentQuality = Math.max(MIN_JPEG_QUALITY, currentQuality - 0.1);

			continue;
		}

		/*
		 * If quality is already low, reduce dimensions.
		 */
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

/* ---------------------------------------------------------------------- */
/* PNG <= 1 MB                                                            */
/* ---------------------------------------------------------------------- */

async function encodePngUnderLimit(img, width, height, enhance) {
	let currentWidth = width;
	let currentHeight = height;

	let bestBlob = null;
	let bestWidth = width;
	let bestHeight = height;

	/*
	 * PNG does not have a quality setting that reliably reduces file size.
	 *
	 * Therefore dimensions are progressively reduced.
	 */
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

		/*
		 * First reduction is moderate.
		 * Later reductions are stronger.
		 */
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

/* ---------------------------------------------------------------------- */
/* MAIN IMAGE PROCESSOR                                                   */
/* ---------------------------------------------------------------------- */

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

	/* ------------------------------------------------------------------ */
	/* AI BACKGROUND REMOVAL                                               */
	/* ------------------------------------------------------------------ */

	if (settings.removeBg) {
		try {
			const aiBlob = await removeBackgroundAI(item.workingSrc);

			if (!(aiBlob instanceof Blob)) {
				throw new Error("Background removal returned an invalid image.");
			}

			const aiUrl = URL.createObjectURL(aiBlob);

			try {
				const transparentImg = await loadImageEl(aiUrl);

				/*
				 * If background removal is the only operation,
				 * preserve the AI output exactly.
				 *
				 * This avoids unnecessary re-rendering and prevents
				 * additional edge degradation.
				 */
				if (!settings.optimize && !settings.enhance) {
					return {
						blob: aiBlob,
						mime: "image/png",
						width: transparentImg.naturalWidth,
						height: transparentImg.naturalHeight,
					};
				}

				/*
				 * Determine final output dimensions.
				 */
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

				/*
				 * Transparent images must remain PNG.
				 */
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

	/* ------------------------------------------------------------------ */
	/* NORMAL IMAGE OPTIMIZATION                                           */
	/* ------------------------------------------------------------------ */

	if (settings.optimize || settings.enhance) {
		return await encodeJpegUnderLimit(
			img,
			w,
			h,
			settings.quality,
			settings.enhance,
		);
	}

	/* ------------------------------------------------------------------ */
	/* NO PROCESSING                                                       */
	/* ------------------------------------------------------------------ */

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

/* ---------------------------------------------------------------------- */
/* COMPONENT                                                              */
/* ---------------------------------------------------------------------- */

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

	const fileInputRef = useRef(null);

	/* ------------------------------------------------------------------ */
	/* CLEANUP                                                             */
	/* ------------------------------------------------------------------ */

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

	/* ------------------------------------------------------------------ */
	/* ADD FILES                                                           */
	/* ------------------------------------------------------------------ */

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

	/* ------------------------------------------------------------------ */
	/* ADD URL                                                             */
	/* ------------------------------------------------------------------ */

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
		} catch {
			// Keep default.
		}

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
			/*
			 * Direct image fallback.
			 */
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

	/* ------------------------------------------------------------------ */
	/* REMOVE ITEM                                                         */
	/* ------------------------------------------------------------------ */

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

	/* ------------------------------------------------------------------ */
	/* CLEAR ALL                                                           */
	/* ------------------------------------------------------------------ */

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

	/* ------------------------------------------------------------------ */
	/* DROP                                                                */
	/* ------------------------------------------------------------------ */

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

	/* ------------------------------------------------------------------ */
	/* PROCESS SEQUENTIAL QUEUE                                            */
	/* ------------------------------------------------------------------ */

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
			/*
			 * Sequential processing is intentional.
			 *
			 * Do NOT use Promise.all() here because multiple ONNX
			 * sessions/images can consume several hundred MB of RAM.
			 */
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
					console.log(`[IMAGE ${index + 1}/${current.length}] ${item.name}`);

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

	/* ------------------------------------------------------------------ */
	/* DOWNLOAD                                                            */
	/* ------------------------------------------------------------------ */

const handleDownload = useCallback(() => {
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

	/* --------------------------------------------------------------- */
	/* SINGLE IMAGE                                                     */
	/* --------------------------------------------------------------- */

	if (done.length === 1) {
		const it = done[0];

		const ext = it.processed.mime === "image/png" ? "png" : "jpg";

		const filename = `${stripExt(it.name)}-optimized.${ext}`;

		downloadBlob(it.processed.blob, filename);

		return;
	}

	/* --------------------------------------------------------------- */
	/* MULTIPLE IMAGES -> ZIP                                          */
	/* --------------------------------------------------------------- */

	(async () => {
		try {
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
					console.warn(`Skipping empty file: ${filename}`);

					continue;
				}

				entries.push({
					name: filename,
					data,
				});

				await yieldToBrowser();
			}

			if (!entries.length) {
				throw new Error(
					"No valid processed images were available for the ZIP.",
				);
			}

			console.log(
				"[ZIP] Creating archive:",
				entries.map((entry) => ({
					name: entry.name,
					size: entry.data.length,
				})),
			);

			const zipBytes = buildZip(entries);

			console.log("[ZIP] Archive created:", formatBytes(zipBytes.length));

			/*
			 * Verify ZIP signature before downloading.
			 *
			 * ZIP files must start with:
			 *
			 * 50 4B 03 04
			 */
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

			downloadBlob(zipBlob, "optimized-images.zip");
		} catch (error) {
			console.error("[ZIP] Failed to create ZIP:", error);

			alert(error?.message || "Unable to create ZIP archive.");
		}
	})();
}, [items]);

	/* ------------------------------------------------------------------ */
	/* STATS                                                               */
	/* ------------------------------------------------------------------ */

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

	/* ------------------------------------------------------------------ */
	/* COLORS                                                              */
	/* ------------------------------------------------------------------ */

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

	/* ------------------------------------------------------------------ */
	/* UI                                                                  */
	/* ------------------------------------------------------------------ */

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

			{/* HEADER */}

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

			{/* INTAKE */}

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

			{/* LOADED IMAGES */}

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

			{/* SETTINGS */}

			<div
				style={{
					background: colors.panel,
					border: `1px solid ${colors.border}`,
					borderRadius: 12,
				}}
				className="mt-6 p-4">
				<div className="flex flex-wrap gap-5">
					{/* REMOVE BG */}

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

					{/* OPTIMIZE */}

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

					{/* ENHANCE */}

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
						AI Processing may take several seconds per image. Works best on images with a clear subject and background.
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
					{/* MAX DIMENSION */}

					<div>
						<div
							className="flex justify-between text-xs mb-1"
							style={{
								color: colors.textDim,
							}}>
							<span>Max dimension</span>

							<span className="mono">{maxDimension}px</span>
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

					{/* JPEG QUALITY */}

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

				{/* SIZE TARGET */}

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

			{/* ACTIONS */}

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

				<button
					onClick={handleDownload}
					disabled={!doneItems.length || processing}
					style={{
						background: colors.panelAlt,
						border: `1px solid ${colors.border}`,
						color: colors.text,
					}}
					className="px-5 py-2.5 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-40">
					<Download size={16} />

					{doneItems.length > 1
						? `Download ${doneItems.length} images (.zip)`
						: "Download image"}
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

			{/* ERROR */}

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

			{/* SAVINGS */}

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

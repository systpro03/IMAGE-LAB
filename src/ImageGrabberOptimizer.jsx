import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
} from "react";

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

import {
  removeBackground as aiRemoveBackground,
  preload as preloadBackgroundRemoval,
} from "@imgly/background-removal";

/* ---------------------------------------------------------------------- */
/* ZIP HELPERS                                                            */
/* ---------------------------------------------------------------------- */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);

  for (let n = 0; n < 256; n++) {
    let c = n;

    for (let k = 0; k < 8; k++) {
      c =
        c & 1
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
    crc =
      CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^
      (crc >>> 8);
  }

  return (crc ^ 0xffffffff) >>> 0;
}

function dosDateTime(d) {
  const date =
    ((d.getFullYear() - 1980) << 9) |
    ((d.getMonth() + 1) << 5) |
    d.getDate();

  const time =
    (d.getHours() << 11) |
    (d.getMinutes() << 5) |
    (d.getSeconds() >> 1);

  return {
    date,
    time,
  };
}

function u16(n) {
  return new Uint8Array([
    n & 0xff,
    (n >> 8) & 0xff,
  ]);
}

function u32(n) {
  return new Uint8Array([
    n & 0xff,
    (n >> 8) & 0xff,
    (n >> 16) & 0xff,
    (n >> 24) & 0xff,
  ]);
}

function concatBytes(chunks) {
  const total = chunks.reduce(
    (s, c) => s + c.length,
    0,
  );

  const out = new Uint8Array(total);

  let off = 0;

  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }

  return out;
}

function strBytes(s) {
  return new TextEncoder().encode(s);
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

    const local = concatBytes([
      u32(0x04034b50),
      u16(20),
      u16(0),
      u16(0),
      u16(time),
      u16(date),
      u32(crc),
      u32(size),
      u32(size),
      u16(nameBytes.length),
      u16(0),
      nameBytes,
      data,
    ]);

    localChunks.push(local);

    const central = concatBytes([
      u32(0x02014b50),
      u16(20),
      u16(20),
      u16(0),
      u16(0),
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

    centralChunks.push(central);

    offset += local.length;
  }

  const centralDir = concatBytes(centralChunks);

  const end = concatBytes([
    u32(0x06054b50),
    u16(0),
    u16(0),
    u16(entries.length),
    u16(entries.length),
    u32(centralDir.length),
    u32(offset),
    u16(0),
  ]);

  return concatBytes([
    ...localChunks,
    centralDir,
    end,
  ]);
}

/* ---------------------------------------------------------------------- */
/* HELPERS                                                                */
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

  return i > 0
    ? name.slice(0, i)
    : name;
}

function loadImageEl(src, crossOrigin) {
  return new Promise((resolve, reject) => {
    const img = new Image();

    if (crossOrigin) {
      img.crossOrigin = crossOrigin;
    }

    img.onload = () => resolve(img);

    img.onerror = () =>
      reject(
        new Error(
          "Couldn't load image.",
        ),
      );

    img.src = src;
  });
}

function downloadBlob(blob, filename) {
  const url =
    URL.createObjectURL(blob);

  const a =
    document.createElement("a");

  a.href = url;
  a.download = filename;

  document.body.appendChild(a);

  a.click();

  a.remove();

  setTimeout(
    () => URL.revokeObjectURL(url),
    4000,
  );
}

/* ---------------------------------------------------------------------- */
/* IMAGE LIMITS                                                           */
/* ---------------------------------------------------------------------- */

const MAX_WORK_DIM = 3000;

/*
 * Maximum dimension sent to the AI model.
 *
 * This is intentionally lower than the final output dimension.
 * Background segmentation does not need an 8000px source image.
 */
const MAX_AI_DIMENSION = 2048;

const MAX_OUTPUT_BYTES =
  1024 * 1024;

const MIN_JPEG_QUALITY = 0.45;

/* ---------------------------------------------------------------------- */
/* AI BACKGROUND REMOVAL                                                  */
/* ---------------------------------------------------------------------- */

let backgroundRemovalReady = false;
let backgroundRemovalLoading = null;

/*
 * Full ISNet gives better segmentation quality than fp16 in cases where
 * hair, clothing and thin edges matter.
 *
 * GPU is attempted first.
 *
 * proxyToWorker keeps heavy work away from the React/UI thread when
 * supported by the installed @imgly/background-removal version.
 */
const AI_BG_CONFIG = {
  model: "isnet",

  device: "gpu",

  proxyToWorker: true,

  output: {
    format: "image/png",
    type: "foreground",
    quality: 1,
  },
};

async function prepareBackgroundRemoval() {
  if (backgroundRemovalReady) {
    return true;
  }

  if (backgroundRemovalLoading) {
    await backgroundRemovalLoading;
    return backgroundRemovalReady;
  }

  backgroundRemovalLoading =
    (async () => {
      try {
        await preloadBackgroundRemoval(
          AI_BG_CONFIG,
        );

        backgroundRemovalReady = true;

        return true;
      } catch (error) {
        /*
         * Do not permanently cache a failed preload.
         *
         * The actual removeBackground call gets another chance.
         */
        backgroundRemovalReady = false;
        backgroundRemovalLoading = null;

        console.warn(
          "Background removal preload failed:",
          error,
        );

        return false;
      }
    })();

  await backgroundRemovalLoading;

  return backgroundRemovalReady;
}

async function removeBackgroundAI(
  src,
  onProgress,
) {
  /*
   * Try preload, but don't fail immediately if preload fails.
   */
  await prepareBackgroundRemoval().catch(
    () => false,
  );

  try {
    const blob =
      await aiRemoveBackground(src, {
        ...AI_BG_CONFIG,

        progress: (
          key,
          current,
          total,
        ) => {
          if (
            typeof onProgress !==
            "function"
          ) {
            return;
          }

          const percent =
            total > 0
              ? Math.round(
                  (current / total) *
                    100,
                )
              : 0;

          onProgress({
            key,
            current,
            total,
            percent,
          });
        },
      });

    if (!(blob instanceof Blob)) {
      throw new Error(
        "Background removal returned an invalid image.",
      );
    }

    return blob;
  } catch (gpuError) {
    /*
     * WebGPU may fail on older browsers or unsupported hardware.
     *
     * Retry using CPU.
     */
    console.warn(
      "GPU background removal failed. Retrying with CPU...",
      gpuError,
    );

    try {
      const blob =
        await aiRemoveBackground(src, {
          ...AI_BG_CONFIG,

          device: "cpu",

          progress: (
            key,
            current,
            total,
          ) => {
            if (
              typeof onProgress !==
              "function"
            ) {
              return;
            }

            const percent =
              total > 0
                ? Math.round(
                    (current / total) *
                      100,
                  )
                : 0;

            onProgress({
              key,
              current,
              total,
              percent,
            });
          },
        });

      if (!(blob instanceof Blob)) {
        throw new Error(
          "CPU background removal returned an invalid image.",
        );
      }

      return blob;
    } catch (cpuError) {
      console.error(
        "CPU background removal failed:",
        cpuError,
      );

      throw new Error(
        "AI background removal failed. The browser may not have enough memory or the AI runtime is unavailable.",
      );
    }
  }
}

/* ---------------------------------------------------------------------- */
/* IMAGE DIMENSION HELPERS                                                */
/* ---------------------------------------------------------------------- */

function getScaledDimensions(
  width,
  height,
  maxDimension,
) {
  const largest =
    Math.max(width, height);

  if (largest <= maxDimension) {
    return {
      width,
      height,
    };
  }

  const scale =
    maxDimension / largest;

  return {
    width: Math.max(
      1,
      Math.round(width * scale),
    ),
    height: Math.max(
      1,
      Math.round(height * scale),
    ),
  };
}

/* ---------------------------------------------------------------------- */
/* AI INPUT PREPARATION                                                   */
/* ---------------------------------------------------------------------- */

async function prepareAIInput(
  src,
  maxDimension = MAX_AI_DIMENSION,
) {
  const img =
    await loadImageEl(src);

  const originalWidth =
    img.naturalWidth;

  const originalHeight =
    img.naturalHeight;

  const {
    width,
    height,
  } = getScaledDimensions(
    originalWidth,
    originalHeight,
    maxDimension,
  );

  /*
   * Already small enough.
   */
  if (
    width === originalWidth &&
    height === originalHeight
  ) {
    return {
      source: src,
      width,
      height,
      originalWidth,
      originalHeight,
      objectUrl: null,
    };
  }

  /*
   * Resize ONLY the AI input.
   */
  const canvas =
    document.createElement(
      "canvas",
    );

  canvas.width = width;
  canvas.height = height;

  const ctx =
    canvas.getContext("2d", {
      alpha: false,
    });

  if (!ctx) {
    throw new Error(
      "Unable to create AI input canvas.",
    );
  }

  ctx.imageSmoothingEnabled =
    true;

  ctx.imageSmoothingQuality =
    "high";

  ctx.drawImage(
    img,
    0,
    0,
    width,
    height,
  );

  const blob =
    await canvasToBlob(
      canvas,
      "image/jpeg",
      0.92,
    );

  /*
   * Release the canvas backing store.
   */
  canvas.width = 1;
  canvas.height = 1;

  const objectUrl =
    URL.createObjectURL(blob);

  return {
    source: objectUrl,
    width,
    height,
    originalWidth,
    originalHeight,
    objectUrl,
  };
}

/* ---------------------------------------------------------------------- */
/* CANVAS HELPERS                                                         */
/* ---------------------------------------------------------------------- */

function drawEnhancedImage(
  img,
  width,
  height,
  {
    transparent = false,
    enhance = false,
  } = {},
) {
  const canvas =
    document.createElement(
      "canvas",
    );

  canvas.width = width;
  canvas.height = height;

  const ctx =
    canvas.getContext("2d", {
      alpha: true,
      willReadFrequently: false,
    });

  if (!ctx) {
    throw new Error(
      "Unable to create image canvas.",
    );
  }

  ctx.clearRect(
    0,
    0,
    width,
    height,
  );

  ctx.imageSmoothingEnabled =
    true;

  ctx.imageSmoothingQuality =
    "high";

  /*
   * IMPORTANT:
   *
   * Do not run the secondary sharpening pass on transparent images.
   *
   * That pass can alter semi-transparent alpha edge pixels and cause
   * white/pale halos around hair, clothing and objects.
   */
  if (
    enhance &&
    !transparent
  ) {
    ctx.filter =
      "contrast(1.035) saturate(1.02) brightness(1.003)";
  } else {
    ctx.filter = "none";
  }

  ctx.globalAlpha = 1;

  ctx.globalCompositeOperation =
    "source-over";

  ctx.drawImage(
    img,
    0,
    0,
    width,
    height,
  );

  ctx.filter = "none";
  ctx.globalAlpha = 1;

  return canvas;
}

/* ---------------------------------------------------------------------- */
/* CANVAS -> BLOB                                                         */
/* ---------------------------------------------------------------------- */

function canvasToBlob(
  canvas,
  mime,
  quality,
) {
  return new Promise(
    (resolve, reject) => {
      try {
        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(
                new Error(
                  "Image encoding failed.",
                ),
              );
            }
          },
          mime,
          quality,
        );
      } catch (error) {
        reject(error);
      }
    },
  );
}

/* ---------------------------------------------------------------------- */
/* JPEG ≤ 1 MB                                                            */
/* ---------------------------------------------------------------------- */

async function encodeJpegUnderLimit(
  img,
  width,
  height,
  quality,
  enhance,
) {
  let currentWidth = width;
  let currentHeight = height;

  let currentQuality = Math.min(
    1,
    Math.max(
      MIN_JPEG_QUALITY,
      quality,
    ),
  );

  let bestBlob = null;

  /*
   * Fewer attempts.
   *
   * The old code could repeatedly create and encode very large canvases.
   */
  for (
    let attempt = 0;
    attempt < 9;
    attempt++
  ) {
    const canvas =
      drawEnhancedImage(
        img,
        currentWidth,
        currentHeight,
        {
          enhance,
          transparent: false,
        },
      );

    const blob =
      await canvasToBlob(
        canvas,
        "image/jpeg",
        currentQuality,
      );

    /*
     * Release canvas backing memory.
     */
    canvas.width = 1;
    canvas.height = 1;

    if (
      !bestBlob ||
      blob.size < bestBlob.size
    ) {
      bestBlob = blob;
    }

    if (
      blob.size <=
      MAX_OUTPUT_BYTES
    ) {
      return {
        blob,
        width: currentWidth,
        height: currentHeight,
        mime: "image/jpeg",
      };
    }

    if (
      currentQuality >
      MIN_JPEG_QUALITY
    ) {
      currentQuality =
        Math.max(
          MIN_JPEG_QUALITY,
          currentQuality - 0.08,
        );

      continue;
    }

    currentWidth =
      Math.max(
        320,
        Math.round(
          currentWidth * 0.84,
        ),
      );

    currentHeight =
      Math.max(
        320,
        Math.round(
          currentHeight * 0.84,
        ),
      );

    currentQuality =
      Math.min(
        0.82,
        quality,
      );

    /*
     * Yield to browser.
     */
    await new Promise(
      (resolve) =>
        setTimeout(resolve, 0),
    );
  }

  return {
    blob: bestBlob,
    width: currentWidth,
    height: currentHeight,
    mime: "image/jpeg",
  };
}

/* ---------------------------------------------------------------------- */
/* PNG ≤ 1 MB                                                             */
/* ---------------------------------------------------------------------- */

async function encodePngUnderLimit(
  img,
  width,
  height,
  enhance,
) {
  let currentWidth = width;
  let currentHeight = height;

  let bestBlob = null;

  /*
   * PNG has no quality parameter.
   *
   * Therefore resizing is the only reliable way to reduce size.
   */
  for (
    let attempt = 0;
    attempt < 8;
    attempt++
  ) {
    const canvas =
      drawEnhancedImage(
        img,
        currentWidth,
        currentHeight,
        {
          enhance: false,
          /*
           * ALWAYS true for AI background removal.
           *
           * Prevents whitening of alpha edges.
           */
          transparent: true,
        },
      );

    const blob =
      await canvasToBlob(
        canvas,
        "image/png",
        1,
      );

    canvas.width = 1;
    canvas.height = 1;

    if (
      !bestBlob ||
      blob.size < bestBlob.size
    ) {
      bestBlob = blob;
    }

    if (
      blob.size <=
      MAX_OUTPUT_BYTES
    ) {
      return {
        blob,
        width: currentWidth,
        height: currentHeight,
        mime: "image/png",
      };
    }

    currentWidth =
      Math.max(
        640,
        Math.round(
          currentWidth * 0.82,
        ),
      );

    currentHeight =
      Math.max(
        640,
        Math.round(
          currentHeight * 0.82,
        ),
      );

    /*
     * Give the browser breathing room.
     */
    await new Promise(
      (resolve) =>
        setTimeout(resolve, 0),
    );
  }

  return {
    blob: bestBlob,
    width: currentWidth,
    height: currentHeight,
    mime: "image/png",
  };
}

/* ---------------------------------------------------------------------- */
/* MAIN IMAGE PROCESSOR                                                   */
/* ---------------------------------------------------------------------- */

async function processImage(
  item,
  settings,
  onProgress,
) {
  const img =
    await loadImageEl(
      item.workingSrc,
      item.needsCrossOrigin
        ? "anonymous"
        : undefined,
    );

  const originalW =
    img.naturalWidth;

  const originalH =
    img.naturalHeight;

  const outputCap =
    settings.optimize
      ? Math.min(
          settings.maxDimension,
          MAX_WORK_DIM,
        )
      : MAX_WORK_DIM;

  const outputDimensions =
    getScaledDimensions(
      originalW,
      originalH,
      outputCap,
    );

  let w =
    outputDimensions.width;

  let h =
    outputDimensions.height;

  /* ------------------------------------------------------------------ */
  /* BACKGROUND REMOVAL                                                 */
  /* ------------------------------------------------------------------ */

  if (settings.removeBg) {
    let aiInput = null;

    try {
      onProgress?.({
        stage: "Preparing AI",
        percent: 3,
      });

      /*
       * IMPORTANT:
       *
       * Never feed an enormous original image directly into the
       * segmentation model.
       */
      aiInput =
        await prepareAIInput(
          item.workingSrc,
          MAX_AI_DIMENSION,
        );

      onProgress?.({
        stage: "AI segmentation",
        percent: 8,
      });

      const aiBlob =
        await removeBackgroundAI(
          aiInput.source,
          (progress) => {
            const modelPercent =
              progress?.percent || 0;

            onProgress?.({
              stage:
                "AI segmentation",
              percent: Math.min(
                80,
                8 +
                  Math.round(
                    modelPercent *
                      0.72,
                  ),
              ),
            });
          },
        );

      onProgress?.({
        stage: "Decoding mask",
        percent: 83,
      });

      const aiUrl =
        URL.createObjectURL(
          aiBlob,
        );

      try {
        const transparentImg =
          await loadImageEl(
            aiUrl,
          );

        /*
         * If no optimization/enhancement is required,
         * return the AI PNG immediately.
         *
         * This avoids an unnecessary decode -> canvas -> PNG encode.
         */
        if (
          !settings.optimize &&
          !settings.enhance
        ) {
          onProgress?.({
            stage: "Complete",
            percent: 100,
          });

          return {
            blob: aiBlob,
            mime: "image/png",
            width: aiInput.width,
            height: aiInput.height,
          };
        }

        onProgress?.({
          stage: "Optimizing PNG",
          percent: 86,
        });

        const aiOutput =
          getScaledDimensions(
            aiInput.width,
            aiInput.height,
            outputCap,
          );

        w = aiOutput.width;
        h = aiOutput.height;

        const result =
          await encodePngUnderLimit(
            transparentImg,
            w,
            h,
            /*
             * Enhancement is deliberately disabled inside
             * transparent PNG rendering.
             *
             * This prevents white halos.
             */
            false,
          );

        onProgress?.({
          stage: "Complete",
          percent: 100,
        });

        return result;
      } finally {
        URL.revokeObjectURL(
          aiUrl,
        );
      }
    } catch (error) {
      console.error(
        "AI background removal failed:",
        error,
      );

      throw new Error(
        error?.message ||
          "AI background removal failed. Please try the image again.",
      );
    } finally {
      /*
       * Release resized AI input.
       */
      if (
        aiInput?.objectUrl
      ) {
        URL.revokeObjectURL(
          aiInput.objectUrl,
        );
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* NORMAL IMAGE OPTIMIZATION                                         */
  /* ------------------------------------------------------------------ */

  if (
    settings.optimize ||
    settings.enhance
  ) {
    onProgress?.({
      stage: "Optimizing image",
      percent: 25,
    });

    const result =
      await encodeJpegUnderLimit(
        img,
        w,
        h,
        settings.quality,
        settings.enhance,
      );

    onProgress?.({
      stage: "Complete",
      percent: 100,
    });

    return result;
  }

  /* ------------------------------------------------------------------ */
  /* NO PROCESSING                                                      */
  /* ------------------------------------------------------------------ */

  const canvas =
    document.createElement(
      "canvas",
    );

  canvas.width = w;
  canvas.height = h;

  const ctx =
    canvas.getContext("2d");

  if (!ctx) {
    throw new Error(
      "Unable to create image canvas.",
    );
  }

  ctx.drawImage(
    img,
    0,
    0,
    w,
    h,
  );

  const mime =
    "image/jpeg";

  const blob =
    await canvasToBlob(
      canvas,
      mime,
      settings.quality,
    );

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
  const [items, setItems] =
    useState([]);

  const [isDragging, setIsDragging] =
    useState(false);

  const [urlInput, setUrlInput] =
    useState("");

  const [processing, setProcessing] =
    useState(false);

  const [removeBg, setRemoveBg] =
    useState(false);

  const [optimize, setOptimize] =
    useState(true);

  const [enhance, setEnhance] =
    useState(true);

  const [quality, setQuality] =
    useState(0.82);

  const [maxDimension, setMaxDimension] =
    useState(1600);

  const fileInputRef =
    useRef(null);

  /* ------------------------------------------------------------------ */
  /* CLEANUP                                                            */
  /* ------------------------------------------------------------------ */

  useEffect(() => {
    return () => {
      items.forEach((it) => {
        if (it.objectUrl) {
          URL.revokeObjectURL(
            it.objectUrl,
          );
        }

        if (it.processed?.url) {
          URL.revokeObjectURL(
            it.processed.url,
          );
        }
      });
    };
  }, []);

  /*
   * IMPORTANT:
   *
   * There is intentionally NO automatic model preload here.
   *
   * The previous useEffect that called prepareBackgroundRemoval()
   * whenever removeBg changed has been removed.
   */

  /* ------------------------------------------------------------------ */
  /* ADD FILES                                                          */
  /* ------------------------------------------------------------------ */

  const addFiles =
    useCallback((fileList) => {
      const files =
        Array.from(
          fileList,
        ).filter((f) =>
          f.type.startsWith(
            "image/",
          ),
        );

      if (!files.length) {
        return;
      }

      setItems((prev) => [
        ...prev,

        ...files.map((file) => {
          const objectUrl =
            URL.createObjectURL(
              file,
            );

          return {
            id: nextId++,
            name: file.name,
            originalSize:
              file.size,
            objectUrl,
            workingSrc:
              objectUrl,
            needsCrossOrigin:
              false,
            loadStatus:
              "ready",
            loadError: null,
            processStatus:
              "idle",
            processProgress: 0,
            processStage: "",
            processError: null,
            processed: null,
          };
        }),
      ]);
    }, []);

  /* ------------------------------------------------------------------ */
  /* ADD URL                                                            */
  /* ------------------------------------------------------------------ */

  const addUrl =
    useCallback(
      async (rawUrl) => {
        const url =
          rawUrl.trim();

        if (!url) {
          return;
        }

        let name = "image";

        try {
          const u =
            new URL(url);

          name =
            decodeURIComponent(
              u.pathname
                .split("/")
                .filter(Boolean)
                .pop() ||
                "image",
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
            needsCrossOrigin:
              false,
            loadStatus:
              "loading",
            loadError: null,
            processStatus:
              "idle",
            processProgress: 0,
            processStage: "",
            processError: null,
            processed: null,
          },
        ]);

        try {
          const res =
            await fetch(url, {
              mode: "cors",
            });

          if (!res.ok) {
            throw new Error(
              "Request failed",
            );
          }

          const blob =
            await res.blob();

          if (
            !blob.type.startsWith(
              "image/",
            )
          ) {
            throw new Error(
              "Not an image",
            );
          }

          const objectUrl =
            URL.createObjectURL(
              blob,
            );

          setItems((prev) =>
            prev.map((it) =>
              it.id === id
                ? {
                    ...it,
                    objectUrl,
                    workingSrc:
                      objectUrl,
                    needsCrossOrigin:
                      false,
                    originalSize:
                      blob.size,
                    loadStatus:
                      "ready",
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
                    workingSrc:
                      url,
                    needsCrossOrigin:
                      true,
                    loadStatus:
                      "ready",
                    loadError:
                      "cors-risk",
                  }
                : it,
            ),
          );
        }
      },
      [],
    );

  /* ------------------------------------------------------------------ */
  /* REMOVE ITEM                                                        */
  /* ------------------------------------------------------------------ */

  const removeItem =
    useCallback((id) => {
      setItems((prev) => {
        const it =
          prev.find(
            (x) => x.id === id,
          );

        if (it?.objectUrl) {
          URL.revokeObjectURL(
            it.objectUrl,
          );
        }

        if (it?.processed?.url) {
          URL.revokeObjectURL(
            it.processed.url,
          );
        }

        return prev.filter(
          (x) => x.id !== id,
        );
      });
    }, []);

  /* ------------------------------------------------------------------ */
  /* CLEAR ALL                                                          */
  /* ------------------------------------------------------------------ */

  const clearAll =
    useCallback(() => {
      items.forEach((it) => {
        if (it.objectUrl) {
          URL.revokeObjectURL(
            it.objectUrl,
          );
        }

        if (it.processed?.url) {
          URL.revokeObjectURL(
            it.processed.url,
          );
        }
      });

      setItems([]);
    }, [items]);

  /* ------------------------------------------------------------------ */
  /* DROP                                                               */
  /* ------------------------------------------------------------------ */

  const handleDrop =
    useCallback(
      (e) => {
        e.preventDefault();

        setIsDragging(false);

        if (
          e.dataTransfer.files &&
          e.dataTransfer.files
            .length
        ) {
          addFiles(
            e.dataTransfer.files,
          );

          return;
        }

        const uri =
          e.dataTransfer.getData(
            "text/uri-list",
          ) ||
          e.dataTransfer.getData(
            "text/plain",
          );

        if (
          uri &&
          /^https?:\/\//i.test(
            uri.trim(),
          )
        ) {
          addUrl(uri);
        }
      },
      [addFiles, addUrl],
    );

  /* ------------------------------------------------------------------ */
  /* PROCESS                                                            */
  /* ------------------------------------------------------------------ */

  const handleProcess =
    useCallback(async () => {
      if (
        !removeBg &&
        !optimize &&
        !enhance
      ) {
        return;
      }

      if (!items.length) {
        return;
      }

      setProcessing(true);

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
          processStatus:
            "processing",
          processProgress: 0,
          processStage:
            "Waiting…",
          processError: null,
        })),
      );

      try {
        /*
         * IMPORTANT:
         *
         * NEVER use Promise.all() for AI background removal.
         *
         * One image at a time prevents the browser from being
         * overwhelmed by multiple ONNX/WASM/WebGPU jobs.
         */
        for (const item of items) {
          try {
            setItems((prev) =>
              prev.map((it) =>
                it.id === item.id
                  ? {
                      ...it,
                      processStatus:
                        "processing",
                      processProgress: 0,
                      processStage:
                        "Starting…",
                    }
                  : it,
              ),
            );

            const result =
              await processImage(
                item,
                settings,
                (progress) => {
                  setItems(
                    (prev) =>
                      prev.map(
                        (it) =>
                          it.id ===
                          item.id
                            ? {
                                ...it,
                                processProgress:
                                  progress?.percent ||
                                  0,
                                processStage:
                                  progress?.stage ||
                                  "Processing…",
                              }
                            : it,
                      ),
                  );
                },
              );

            const url =
              URL.createObjectURL(
                result.blob,
              );

            setItems((prev) =>
              prev.map((it) => {
                if (
                  it.id !==
                  item.id
                ) {
                  return it;
                }

                if (
                  it.processed?.url
                ) {
                  URL.revokeObjectURL(
                    it.processed.url,
                  );
                }

                return {
                  ...it,
                  processStatus:
                    "done",
                  processProgress:
                    100,
                  processStage:
                    "Complete",
                  processed: {
                    blob:
                      result.blob,
                    url,
                    size:
                      result.blob
                        .size,
                    width:
                      result.width,
                    height:
                      result.height,
                    mime:
                      result.mime,
                  },
                };
              }),
            );

            /*
             * Let the browser render the result and release
             * memory before starting the next image.
             */
            await new Promise(
              (resolve) =>
                setTimeout(
                  resolve,
                  75,
                ),
            );
          } catch (err) {
            console.error(
              `Failed processing ${item.name}:`,
              err,
            );

            setItems((prev) =>
              prev.map((it) =>
                it.id === item.id
                  ? {
                      ...it,
                      processStatus:
                        "error",
                      processProgress: 0,
                      processStage:
                        "Failed",
                      processError:
                        err?.message ||
                        "Processing failed.",
                    }
                  : it,
              ),
            );

            /*
             * Continue with the next image.
             */
          }
        }
      } finally {
        setProcessing(false);
      }
    }, [
      items,
      removeBg,
      optimize,
      enhance,
      quality,
      maxDimension,
    ]);

  /* ------------------------------------------------------------------ */
  /* DOWNLOAD                                                           */
  /* ------------------------------------------------------------------ */

  const handleDownload =
    useCallback(() => {
      const done =
        items.filter(
          (it) =>
            it.processStatus ===
              "done" &&
            it.processed,
        );

      if (!done.length) {
        return;
      }

      if (done.length === 1) {
        const it = done[0];

        const ext =
          it.processed.mime ===
          "image/png"
            ? "png"
            : "jpg";

        downloadBlob(
          it.processed.blob,
          `${stripExt(
            it.name,
          )}-optimized.${ext}`,
        );

        return;
      }

      (async () => {
        const used = new Set();
        const entries = [];

        for (const it of done) {
          const ext =
            it.processed.mime ===
            "image/png"
              ? "png"
              : "jpg";

          let name = `${stripExt(
            it.name,
          )}-optimized.${ext}`;

          let n = 2;

          while (
            used.has(name)
          ) {
            name = `${stripExt(
              it.name,
            )}-optimized (${n}).${ext}`;

            n++;
          }

          used.add(name);

          const buf =
            await it.processed.blob.arrayBuffer();

          entries.push({
            name,
            data:
              new Uint8Array(
                buf,
              ),
          });
        }

        const zipBytes =
          buildZip(entries);

        downloadBlob(
          new Blob(
            [zipBytes],
            {
              type: "application/zip",
            },
          ),
          "optimized-images.zip",
        );
      })();
    }, [items]);

  /* ------------------------------------------------------------------ */
  /* STATS                                                              */
  /* ------------------------------------------------------------------ */

  const doneItems =
    items.filter(
      (it) =>
        it.processStatus ===
          "done" &&
        it.processed,
    );

  const errorItems =
    items.filter(
      (it) =>
        it.processStatus ===
        "error",
    );

  const totalOriginal =
    doneItems.reduce(
      (s, it) =>
        s +
        (it.originalSize ||
          0),
      0,
    );

  const totalProcessed =
    doneItems.reduce(
      (s, it) =>
        s +
        (it.processed?.size ||
          0),
      0,
    );

  const savedPct =
    totalOriginal > 0
      ? Math.max(
          0,
          Math.round(
            (1 -
              totalProcessed /
                totalOriginal) *
              100,
          ),
        )
      : null;

  const overLimitCount =
    doneItems.filter(
      (it) =>
        it.processed.size >
        MAX_OUTPUT_BYTES,
    ).length;

  /* ------------------------------------------------------------------ */
  /* COLORS                                                             */
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
  /* UI                                                                 */
  /* ------------------------------------------------------------------ */

  return (
    <div
      style={{
        background:
          colors.bg,
        color:
          colors.text,
        minHeight:
          "100%",
      }}
      className="w-full rounded-xl p-6 font-sans"
    >
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
            color:
              colors.accent,
          }}
        >
          Image Lab
        </div>

        <h1 className="text-2xl font-bold tracking-tight">
          Grabber, background remover &amp;
          optimizer
        </h1>

        <p
          className="text-sm mt-1"
          style={{
            color:
              colors.textDim,
          }}
        >
          Drop an image link or browse local
          files, then enhance, remove the
          background and/or compress before
          downloading.
        </p>
      </div>

      {/* INTAKE */}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() =>
          setIsDragging(false)
        }
        onDrop={handleDrop}
        style={{
          border: `1.5px dashed ${
            isDragging
              ? colors.accent
              : colors.border
          }`,

          background:
            isDragging
              ? colors.accentDim +
                "33"
              : colors.panel,

          borderRadius: 12,
        }}
        className="p-8 flex flex-col items-center justify-center text-center transition-colors"
      >
        <Upload
          size={28}
          style={{
            color:
              colors.textDim,
          }}
          className="mb-3"
        />

        <p className="text-sm mb-1">
          Drag image files or an image link
          here
        </p>

        <p
          className="text-xs mb-4"
          style={{
            color:
              colors.textDim,
          }}
        >
          or
        </p>

        <button
          onClick={() =>
            fileInputRef.current?.click()
          }
          style={{
            background:
              colors.panelAlt,
            border: `1px solid ${colors.border}`,
            color:
              colors.text,
          }}
          className="px-4 py-2 rounded-lg text-sm flex items-center gap-2 hover:brightness-110"
        >
          <FolderOpen
            size={16}
          />
          Browse local files
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            if (
              e.target.files
            ) {
              addFiles(
                e.target.files,
              );
            }

            e.target.value =
              "";
          }}
        />

        <div className="flex items-center gap-2 mt-5 w-full max-w-md">
          <Link2
            size={16}
            style={{
              color:
                colors.textDim,
            }}
          />

          <input
            value={urlInput}
            onChange={(e) =>
              setUrlInput(
                e.target.value,
              )
            }
            onKeyDown={(e) => {
              if (
                e.key ===
                  "Enter" &&
                urlInput.trim()
              ) {
                addUrl(
                  urlInput,
                );

                setUrlInput(
                  "",
                );
              }
            }}
            placeholder="https://example.com/photo.jpg"
            style={{
              background:
                colors.panelAlt,
              border: `1px solid ${colors.border}`,
              color:
                colors.text,
            }}
            className="flex-1 rounded-lg px-3 py-2 text-sm outline-none"
          />

          <button
            onClick={() => {
              if (
                urlInput.trim()
              ) {
                addUrl(
                  urlInput,
                );

                setUrlInput(
                  "",
                );
              }
            }}
            style={{
              background:
                colors.panelAlt,
              border: `1px solid ${colors.border}`,
              color:
                colors.text,
            }}
            className="px-3 py-2 rounded-lg text-sm hover:brightness-110"
          >
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
                color:
                  colors.textDim,
              }}
            >
              {items.length} image
              {items.length > 1
                ? "s"
                : ""}{" "}
              loaded
            </span>

            <button
              onClick={
                clearAll
              }
              style={{
                color:
                  colors.textDim,
              }}
              className="text-xs flex items-center gap-1 hover:text-white"
            >
              <Trash2
                size={12}
              />
              Clear all
            </button>
          </div>

          <div
            className="grid gap-3"
            style={{
              gridTemplateColumns:
                "repeat(auto-fill, minmax(150px, 1fr))",
            }}
          >
            {items.map(
              (it) => (
                <div
                  key={it.id}
                  style={{
                    background:
                      colors.panel,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 10,
                  }}
                  className="p-2 relative"
                >
                  <button
                    onClick={() =>
                      removeItem(
                        it.id,
                      )
                    }
                    style={{
                      background:
                        colors.panelAlt,
                      color:
                        colors.textDim,
                    }}
                    className="absolute top-1 right-1 rounded-full p-1 hover:text-white z-10"
                    aria-label={`Remove ${it.name}`}
                  >
                    <X
                      size={12}
                    />
                  </button>

                  <div
                    className="checker rounded-md overflow-hidden mb-2"
                    style={{
                      aspectRatio:
                        "1/1",
                    }}
                  >
                    <img
                      src={
                        it
                          .processed
                          ?.url ||
                        it.workingSrc
                      }
                      alt={
                        it.name
                      }
                      className="w-full h-full object-contain"
                    />
                  </div>

                  <p
                    className="text-xs truncate mb-0.5"
                    title={
                      it.name
                    }
                  >
                    {it.name}
                  </p>

                  <p
                    className="text-[11px] mono"
                    style={{
                      color:
                        colors.textDim,
                    }}
                  >
                    {formatBytes(
                      it.originalSize,
                    )}

                    {it.processed && (
                      <>
                        {" → "}

                        <span
                          style={{
                            color:
                              colors.good,
                          }}
                        >
                          {formatBytes(
                            it
                              .processed
                              .size,
                          )}
                        </span>
                      </>
                    )}
                  </p>

                  {/* PROCESSING */}

                  {it.processStatus ===
                    "processing" && (
                    <div
                      className="mt-1 text-[11px]"
                      style={{
                        color:
                          colors.accent,
                      }}
                    >
                      <div className="flex items-center gap-1">
                        <Loader2
                          size={11}
                          className="animate-spin"
                        />

                        <span>
                          {it.processStage ||
                            "Processing…"}
                        </span>

                        <span className="mono ml-auto">
                          {it.processProgress ||
                            0}
                          %
                        </span>
                      </div>

                      <div
                        className="mt-1 h-1 rounded-full overflow-hidden"
                        style={{
                          background:
                            colors.border,
                        }}
                      >
                        <div
                          style={{
                            width: `${Math.min(
                              100,
                              it.processProgress ||
                                0,
                            )}%`,
                            height:
                              "100%",
                            background:
                              colors.accent,
                            transition:
                              "width 120ms ease",
                          }}
                        />
                      </div>
                    </div>
                  )}

                  {/* ERROR */}

                  {it.processStatus ===
                    "error" && (
                    <div
                      className="flex items-start gap-1 mt-1 text-[11px]"
                      style={{
                        color:
                          colors.bad,
                      }}
                    >
                      <AlertCircle
                        size={11}
                        className="mt-0.5 shrink-0"
                      />

                      <span>
                        {
                          it.processError
                        }
                      </span>
                    </div>
                  )}

                  {/* CORS */}

                  {it.loadError ===
                    "cors-risk" &&
                    it.processStatus ===
                      "idle" && (
                      <div
                        className="flex items-start gap-1 mt-1 text-[11px]"
                        style={{
                          color:
                            colors.textDim,
                        }}
                      >
                        <AlertCircle
                          size={11}
                          className="mt-0.5 shrink-0"
                        />

                        <span>
                          May fail to
                          process —
                          source blocks
                          cross-origin
                          access.
                        </span>
                      </div>
                    )}
                </div>
              ),
            )}
          </div>
        </div>
      )}

      {/* SETTINGS */}

      <div
        style={{
          background:
            colors.panel,
          border: `1px solid ${colors.border}`,
          borderRadius: 12,
        }}
        className="mt-6 p-4"
      >
        <div className="flex flex-wrap gap-5">
          {/* REMOVE BG */}

          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={
                removeBg
              }
              onChange={(e) =>
                setRemoveBg(
                  e.target
                    .checked,
                )
              }
            />

            <Scissors
              size={14}
            />

            Remove background
          </label>

          {/* OPTIMIZE */}

          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={
                optimize
              }
              onChange={(e) =>
                setOptimize(
                  e.target
                    .checked,
                )
              }
            />

            <Wand2
              size={14}
            />

            Optimize &amp;
            compress
          </label>

          {/* ENHANCE */}

          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={
                enhance
              }
              onChange={(e) =>
                setEnhance(
                  e.target
                    .checked,
                )
              }
            />

            <Sparkles
              size={14}
            />

            Enhance image quality
          </label>
        </div>

        {removeBg && (
          <p
            className="text-xs mt-2"
            style={{
              color:
                colors.textDim,
            }}
          >
            Higher-quality AI segmentation
            is used for cleaner hair, clothing,
            fingers and thin edges. Large source
            images are internally resized before
            AI processing to keep the browser
            responsive.
          </p>
        )}

        {enhance && (
          <p
            className="text-xs mt-2"
            style={{
              color:
                colors.textDim,
            }}
          >
            Applies subtle clarity, contrast and
            sharpening to normal images. AI
            transparent PNG edges are not
            sharpened to prevent white halos.
          </p>
        )}

        <div className="grid sm:grid-cols-2 gap-4 mt-4">
          {/* MAX DIMENSION */}

          <div>
            <div
              className="flex justify-between text-xs mb-1"
              style={{
                color:
                  colors.textDim,
              }}
            >
              <span>
                Max dimension
              </span>

              <span className="mono">
                {
                  maxDimension
                }
                px
              </span>
            </div>

            <select
              disabled={!optimize}
              value={
                maxDimension
              }
              onChange={(e) =>
                setMaxDimension(
                  Number(
                    e.target
                      .value,
                  ),
                )
              }
              style={{
                background:
                  colors.panelAlt,
                border: `1px solid ${colors.border}`,
                color:
                  colors.text,
              }}
              className="w-full rounded-lg px-2 py-1.5 text-sm disabled:opacity-40"
            >
              <option value={800}>
                800px
              </option>

              <option value={1280}>
                1280px
              </option>

              <option value={1600}>
                1600px
              </option>

              <option value={2000}>
                2000px
              </option>

              <option value={2500}>
                2500px (original size
                if smaller)
              </option>
            </select>
          </div>

          {/* QUALITY */}

          <div>
            <div
              className="flex justify-between text-xs mb-1"
              style={{
                color:
                  colors.textDim,
              }}
            >
              <span>
                JPEG quality
              </span>

              <span className="mono">
                {Math.round(
                  quality * 100,
                )}
                %
              </span>
            </div>

            <input
              type="range"
              min={0.45}
              max={1}
              step={0.05}
              disabled={
                !optimize ||
                removeBg
              }
              value={quality}
              onChange={(e) =>
                setQuality(
                  Number(
                    e.target
                      .value,
                  ),
                )
              }
              className="w-full disabled:opacity-40"
            />

            {removeBg && (
              <p
                className="text-[11px] mt-1"
                style={{
                  color:
                    colors.textDim,
                }}
              >
                Doesn't apply —
                transparent images
                export as PNG.
              </p>
            )}
          </div>
        </div>

        {/* AI LIMIT */}

        {removeBg && (
          <div
            className="mt-4 rounded-lg px-3 py-2 text-xs"
            style={{
              background:
                colors.panelAlt,
              color:
                colors.textDim,
            }}
          >
            AI processing limit:
            <strong
              className="ml-1"
              style={{
                color:
                  colors.accent,
              }}
            >
              2048px
            </strong>

            <span className="ml-1">
              internally. This prevents
              large images from freezing
              the browser.
            </span>
          </div>
        )}

        {/* SIZE TARGET */}

        <div
          className="mt-4 rounded-lg px-3 py-2 text-xs flex items-center gap-2"
          style={{
            background:
              colors.panelAlt,
            color:
              colors.textDim,
          }}
        >
          <Sparkles
            size={13}
          />

          Output target:

          <strong
            style={{
              color:
                colors.good,
            }}
          >
            ≤ 1 MB
          </strong>

          per image
        </div>
      </div>

      {/* ACTIONS */}

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          onClick={
            handleProcess
          }
          disabled={
            !items.length ||
            processing ||
            (!removeBg &&
              !optimize &&
              !enhance)
          }
          style={{
            background:
              colors.accent,
            color:
              "#211505",
          }}
          className="px-5 py-2.5 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-40"
        >
          {processing ? (
            <Loader2
              size={16}
              className="animate-spin"
            />
          ) : (
            <Wand2
              size={16}
            />
          )}

          {processing
            ? "Processing…"
            : "Process images"}
        </button>

        <button
          onClick={
            handleDownload
          }
          disabled={
            !doneItems.length
          }
          style={{
            background:
              colors.panelAlt,
            border: `1px solid ${colors.border}`,
            color:
              colors.text,
          }}
          className="px-5 py-2.5 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-40"
        >
          <Download
            size={16}
          />

          {doneItems.length >
          1
            ? `Download ${doneItems.length} images (.zip)`
            : "Download image"}
        </button>

        {!removeBg &&
          !optimize &&
          !enhance && (
            <span
              className="text-xs"
              style={{
                color:
                  colors.textDim,
              }}
            >
              Choose at least one
              action above.
            </span>
          )}
      </div>

      {/* ERROR */}

      {errorItems.length >
        0 && (
        <p
          className="text-xs mt-2"
          style={{
            color:
              colors.bad,
          }}
        >
          {errorItems.length} of{" "}
          {items.length} image
          {items.length > 1
            ? "s"
            : ""}{" "}
          failed to process and
          won't be included in the
          download.
        </p>
      )}

      {/* SAVINGS */}

      {doneItems.length >
        0 && (
        <div
          style={{
            background:
              colors.panel,
            border: `1px solid ${colors.border}`,
            borderRadius: 12,
          }}
          className="mt-6 p-4 flex flex-wrap gap-6"
        >
          <div>
            <div
              className="text-[11px] uppercase tracking-widest"
              style={{
                color:
                  colors.textDim,
              }}
            >
              Original
            </div>

            <div className="mono text-lg">
              {formatBytes(
                totalOriginal,
              )}
            </div>
          </div>

          <div>
            <div
              className="text-[11px] uppercase tracking-widest"
              style={{
                color:
                  colors.textDim,
              }}
            >
              Processed
            </div>

            <div
              className="mono text-lg"
              style={{
                color:
                  colors.good,
              }}
            >
              {formatBytes(
                totalProcessed,
              )}
            </div>
          </div>

          {savedPct !==
            null && (
            <div>
              <div
                className="text-[11px] uppercase tracking-widest"
                style={{
                  color:
                    colors.textDim,
                }}
              >
                Saved
              </div>

              <div
                className="mono text-lg"
                style={{
                  color:
                    colors.accent,
                }}
              >
                {savedPct}%
              </div>
            </div>
          )}

          <div>
            <div
              className="text-[11px] uppercase tracking-widest"
              style={{
                color:
                  colors.textDim,
              }}
            >
              Size limit
            </div>

            <div
              className="mono text-lg"
              style={{
                color:
                  overLimitCount >
                  0
                    ? colors.bad
                    : colors.good,
              }}
            >
              {overLimitCount >
              0
                ? `${overLimitCount} over`
                : "≤ 1 MB"}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
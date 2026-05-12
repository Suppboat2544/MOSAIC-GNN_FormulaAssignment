/**
 * Next.js App Router API route — proxies multipart file uploads to FastAPI.
 *
 * The Next.js `rewrites` proxy does not reliably stream binary
 * multipart/form-data, causing EPIPE / ECONNRESET errors when users upload
 * XLSX (or any binary) files. This route reads the FormData in-process and
 * re-posts it directly to the backend with Node's native fetch, which handles
 * binary streams correctly.
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();

    const res = await fetch(`${BACKEND}/api/v1/upload/`, {
      method: "POST",
      body: formData,
      // Do NOT set Content-Type manually — fetch sets it with the correct
      // multipart boundary automatically when body is FormData.
    });

    const json = await res.json();
    return NextResponse.json(json, { status: res.status });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: `Proxy error: ${msg}` }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  // Forwarding GET /api/v1/upload/<job_id>/columns is handled by the rewrite;
  // this stub exists only so Next.js doesn't 405 on accidental GET to this path.
  return NextResponse.json({ detail: "Use POST to upload a file." }, { status: 405 });
}

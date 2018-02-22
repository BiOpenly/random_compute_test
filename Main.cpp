#include <array>
#include <map>
#include <memory>
#include <vector>

#include <ctime>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <epoxy/gl.h>

#include "Context.h"
#include "GLUtils.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#if defined _M_GENERIC
#  define _M_SSE 0
#elif _MSC_VER || __INTEL_COMPILER
#  define _M_SSE 0x402
#elif defined __GNUC__
# if defined __SSE4_2__
#  define _M_SSE 0x402
# elif defined __SSE4_1__
#  define _M_SSE 0x401
# elif defined __SSSE3__
#  define _M_SSE 0x301
# elif defined __SSE3__
#  define _M_SSE 0x300
# endif
#endif


enum class TexType
{
	TYPE_RGB565,
};

#define SSEOPT 0
#if SSEOPT
	void DecodeOnCPU(uint32_t* dst, uint8_t* src, int width, int height, TexType type)
	{
		const int Wsteps4 = (width + 3) / 4;
		const int Wsteps8 = (width + 7) / 8;

		switch(type)
		{
		case TexType::TYPE_RGB565:
			{
				// JSD optimized with SSE2 intrinsics.
				// Produces an ~78% speed improvement over reference C implementation.
				const __m128i kMaskR0 = _mm_set1_epi32(0x000000F8);
				const __m128i kMaskG0 = _mm_set1_epi32(0x0000FC00);
				const __m128i kMaskG1 = _mm_set1_epi32(0x00000300);
				const __m128i kMaskB0 = _mm_set1_epi32(0x00F80000);
				const __m128i kAlpha  = _mm_set1_epi32(0xFF000000);
				for (int y = 0; y < height; y += 4)
					for (int x = 0, yStep = (y / 4) * Wsteps4; x < width; x += 4, yStep++)
						for (int iy = 0, xStep = 4 * yStep; iy < 4; iy++, xStep++)
						{
							__m128i *dxtsrc = (__m128i *)(src + 8 * xStep);
							// Load 4x 16-bit colors: (0000 0000 hgfe dcba)
							// where hg, fe, ba, and dc are 16-bit colors in big-endian order
							const __m128i rgb565x4 = _mm_loadl_epi64(dxtsrc);

							// The big-endian 16-bit colors `ba` and `dc` look like 0b_gggBBBbb_RRRrrGGg in a little endian xmm register
							// Unpack `hgfe dcba` to `hhgg ffee ddcc bbaa`, where each 32-bit word is now 0b_gggBBBbb_RRRrrGGg_gggBBBbb_RRRrrGGg
							const __m128i c0 = _mm_unpacklo_epi16(rgb565x4, rgb565x4);

							// swizzle 0b_gggBBBbb_RRRrrGGg_gggBBBbb_RRRrrGGg
							//      to 0b_11111111_BBBbbBBB_GGggggGG_RRRrrRRR

							// 0b_gggBBBbb_RRRrrGGg_gggBBBbb_RRRrrGGg &
							// 0b_00000000_00000000_00000000_11111000 =
							// 0b_00000000_00000000_00000000_RRRrr000
							const __m128i r0 = _mm_and_si128(c0, kMaskR0);
							// 0b_00000000_00000000_00000000_RRRrr000 >> 5 [32] =
							// 0b_00000000_00000000_00000000_00000RRR
							const __m128i r1 = _mm_srli_epi32(r0, 5);

							// 0b_gggBBBbb_RRRrrGGg_gggBBBbb_RRRrrGGg >> 3 [32] =
							// 0b_000gggBB_BbbRRRrr_GGggggBB_BbbRRRrr &
							// 0b_00000000_00000000_11111100_00000000 =
							// 0b_00000000_00000000_GGgggg00_00000000
							const __m128i gtmp = _mm_srli_epi32(c0, 3);
							const __m128i g0 = _mm_and_si128(gtmp, kMaskG0);
							// 0b_GGggggBB_BbbRRRrr_GGggggBB_Bbb00000 >> 6 [32] =
							// 0b_000000GG_ggggBBBb_bRRRrrGG_ggggBBBb &
							// 0b_00000000_00000000_00000011_00000000 =
							// 0b_00000000_00000000_000000GG_00000000 =
							const __m128i g1 = _mm_and_si128(_mm_srli_epi32(gtmp, 6), kMaskG1);

							// 0b_gggBBBbb_RRRrrGGg_gggBBBbb_RRRrrGGg >> 5 [32] =
							// 0b_00000ggg_BBBbbRRR_rrGGgggg_BBBbbRRR &
							// 0b_00000000_11111000_00000000_00000000 =
							// 0b_00000000_BBBbb000_00000000_00000000
							const __m128i b0 = _mm_and_si128(_mm_srli_epi32(c0, 5), kMaskB0);
							// 0b_00000000_BBBbb000_00000000_00000000 >> 5 [16] =
							// 0b_00000000_00000BBB_00000000_00000000
							const __m128i b1 = _mm_srli_epi16(b0, 5);

							// OR together the final RGB bits and the alpha component:
							const __m128i abgr888x4 = _mm_or_si128(
								_mm_or_si128(
									_mm_or_si128(r0, r1),
									_mm_or_si128(g0, g1)
								),
								_mm_or_si128(
									_mm_or_si128(b0, b1),
									kAlpha
								)
							);

							__m128i *ptr = (__m128i *)(dst + (y + iy) * width + x);
							_mm_storeu_si128(ptr, abgr888x4);
						}
			}

		break;
		}
	}
#else
	constexpr uint8_t Convert3To8(uint8_t v)
	{
		// Swizzle bits: 00000123 -> 12312312
		return (v << 5) | (v << 2) | (v >> 1);
	}

	constexpr uint8_t Convert4To8(uint8_t v)
	{
		// Swizzle bits: 00001234 -> 12341234
		return (v << 4) | v;
	}

	constexpr uint8_t Convert5To8(uint8_t v)
	{
		// Swizzle bits: 00012345 -> 12345123
		return (v << 3) | (v >> 2);
	}

	constexpr uint8_t Convert6To8(uint8_t v)
	{
		// Swizzle bits: 00123456 -> 12345612
		return (v << 2) | (v >> 4);
	}
	static inline uint32_t DecodePixel_RGB565(uint16_t val)
	{
		int r,g,b,a;
		r=Convert5To8((val>>11) & 0x1f);
		g=Convert6To8((val>>5 ) & 0x3f);
		b=Convert5To8((val    ) & 0x1f);
		a=0xFF;
		return  r | (g<<8) | (b << 16) | (a << 24);
	}
#include <byteswap.h>

	inline uint16_t swap16(uint16_t _data) {return bswap_16(_data);}
	inline uint32_t swap32(uint32_t _data) {return bswap_32(_data);}
	inline uint64_t swap64(uint64_t _data) {return bswap_64(_data);}

	void DecodeOnCPU(uint32_t* dst, uint8_t* src, int width, int height, TexType type)
	{
		const int Wsteps4 = (width + 3) / 4;
		const int Wsteps8 = (width + 7) / 8;

		switch(type)
		{
		case TexType::TYPE_RGB565:
			// Reference C implementation.
			for (int y = 0; y < height; y += 4)
				for (int x = 0; x < width; x += 4)
					for (int iy = 0; iy < 4; iy++, src += 8)
					{
						uint32_t *ptr = dst + (y + iy) * width + x;
						uint16_t *s = (uint16_t *)src;
						for (int j = 0; j < 4; j++)
							*ptr++ = DecodePixel_RGB565(swap16(*s++));
					}
		break;
		}
	}
#endif

std::map<TexType, GLuint> s_pgms;
uint32_t TexDim = 0;
GLuint GenerateDecoderProgram(TexType type)
{
	auto it = s_pgms.find(type);
	if (it != s_pgms.end())
		return it->second;

	switch(type)
	{
	case TexType::TYPE_RGB565:
	{
		const char* fs_test =
		"#version 320 es\n"
		"precision highp uimageBuffer;\n"
		"precision highp uimage2D;\n"
		"precision highp float;\n"

		"layout(rgba16ui, binding = 0) readonly uniform uimageBuffer enc_tex;\n"

		"uint Convert5To8(uint val)\n"
		"{\n"
			"\treturn (val << 3) | (val >> 2);\n"
		"}\n\n"

		"uint Convert6To8(uint val)\n"
		"{\n"
			"\treturn (val << 2) | (val >> 4);\n"
		"}\n\n"

		"uint bswap16(uint src)\n"
		"{\n"
		"	return ((src & 0xFFu) << 8u) | (src >> 8u);\n"
		"}\n"

		"uvec4 LoadTexel(ivec2 dim, ivec2 loc)\n"
		"{\n"
			// X and Y are provided in regular linear x/y coordinates
			// Source coordinates are Width * y + x
			// Workgroup size is width / 4
			// So source is (imageDim * (y / 2)) + (x / 2)
			// enc_tex has two pixels per u32 so we need to
			// Each texture load loads a u32
			// Src 0 = (0,0) & (0, 1)
			// Src 1 = (0,2) & (0x3)
			// SRCDIM | offset
			// ---------------
			// (0, 0) | 0 (r)
			// (0, 1) | 0 (g)
			// (0, 2) | 0 (b)
			// (0, 3) | 0 (a)
			// (1, 0) | (32*1 / 2)+0/2 = 16
			// (1, 1) | 16
			// (1, 2) | 17
			"\tint srcloc = ((dim.x * loc.y) >> 2) + (loc.x >> 2);\n"
			"\tuvec4 col0 = imageLoad(enc_tex, srcloc);\n"
			"\tcol0[0] = bswap16(col0[0]);\n"
			"\tcol0[1] = bswap16(col0[1]);\n"
			"\tcol0[2] = bswap16(col0[2]);\n"
			"\tcol0[3] = bswap16(col0[3]);\n"
			"\treturn col0;\n"
		"}\n\n"

		"out vec4 ocol;\n"
		"// RGB565\n"
		"void main() {\n"
		"	ocol = vec4(255);\n"
		"}\n";

		const char* vs_test =
			"#version 320 es\n"
			"out vec2 uv;\n"
			"uniform vec2 src_rect;\n"
			"void main() {\n"
			"	vec2 rawpos = vec2(gl_VertexID & 1, gl_VertexID & 2);\n"
			"	gl_Position = vec4(rawpos*2.0 - 1.0, 0.0, 1.0);\n"
			"	uv = rawpos * src_rect;\n"
			"}\n";

		//for (int y = 0; y < height; y += 4)
		//	for (int x = 0; x < width; x += 4)
		//		for (int iy = 0; iy < 4; iy++, src += 8)
		//		{
		//			uint32_t *ptr = dst + (y + iy) * width + x;
		//			uint16_t *s = (uint16_t *)src;
		//			for (int j = 0; j < 4; j++)
		//				*ptr++ = DecodePixel_RGB565(swap16(*s++));
		//		}

		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		GLuint fs_pgm = glCreateProgram();

		glShaderSource(fs, 1, &fs_test, NULL);
		glShaderSource(vs, 1, &vs_test, NULL);

		glCompileShader(fs);
		glCompileShader(vs);

		GLUtils::CheckShaderStatus(fs, "fs", fs_test);
		GLUtils::CheckShaderStatus(vs, "vs", vs_test);

		glAttachShader(fs_pgm, fs);
		glAttachShader(fs_pgm, vs);
		glLinkProgram(fs_pgm);

		GLUtils::CheckProgramLinkStatus(fs_pgm);
		s_pgms[type] = fs_pgm;
		return fs_pgm;
	}

	break;
	}
}
class GPUTimer
{
public:
	GPUTimer()
	{
		glGenQueries(1, &m_query);
	}
	~GPUTimer()
	{
		glDeleteQueries(1, &m_query);
	}

	void BeginTimer()
	{
		glBeginQuery(GL_TIME_ELAPSED, m_query);
	}

	void EndTimer()
	{
		glEndQuery(GL_TIME_ELAPSED);
	}

	uint64_t GetTime()
	{
		uint64_t res = 0;
		glGetQueryObjectui64v(m_query, GL_QUERY_RESULT, &res);
		return res;
	}
	static int64_t GetTimestamp()
	{
		int64_t res = 0;
		glGetInteger64v(GL_TIMESTAMP, &res);
		return res;
	}

private:
	GLuint m_query;
};

class CPUTimer
{
public:
	CPUTimer()
	{
	}

	void Start()
	{
		start = GetTime();
	}

	uint64_t End()
	{
		end = GetTime();
		return end - start;
	}

	static uint64_t GetTime()
	{
		struct timespec t;
		(void)clock_gettime(CLOCK_MONOTONIC, &t);
		return ((uint64_t)(t.tv_sec * 1000000 + t.tv_nsec / 1000));
	}
private:
	uint64_t start, end;
};

class TextureConvert
{
public:
	TextureConvert(TexType type, int w, int h)
		: m_type(type), m_w(w), m_h(h)
	{
		GLuint imgs[2];

		glGenTextures(2, imgs);
		glGenBuffers(1, &enc_buf);
		enc_img = imgs[0];
		dec_img = imgs[1];

		printf("Creating texture\n");
		// Encoded image
		glBindTexture(GL_TEXTURE_BUFFER, enc_img);
		glBindBuffer(GL_TEXTURE_BUFFER, enc_buf);

		// 8 bits per component
		// 4 components per colour
		// But since we are doing RGB565 it will be 16bits per colour
		// So there will be two colours per texel fetch
		data.resize(m_w * m_h * 2);
		cpudata.resize(m_w * m_h * 4);
		GenRGB565();
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA16UI, enc_buf);

		// Decoded image
		glGenFramebuffers(1, &dec_frame);
		glBindTexture(GL_TEXTURE_2D, dec_img);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_w, m_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, dec_frame);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dec_img, 0);

		mpgm = GenerateDecoderProgram(m_type);
		uRect = glGetUniformLocation(mpgm, "src_rect");

		printf("Done creating\n");

		m_cputime.Start();
		m_avgtime.Start();
	}

	void GenRGB565()
	{
		uint64_t time = m_cputime.End() / 1000;
		if (time >= 2000)
		{
			m_cputime.Start();

			m_shift_val <<= 1;
			if (m_shift_val > m_w)
				m_shift_val = 1;

			for (int y = 0; y < m_h; ++y)
				for (int x = 0; x < m_w; ++x)
				{
					int i = (y * m_w + x) * 2;
					if (x & m_shift_val)
						*(uint16_t*)&data[i] = 0xE0FF;
					else
						*(uint16_t*)&data[i] = 0xFF07;

				}

			glBindBuffer(GL_TEXTURE_BUFFER, enc_buf);
			glBufferData(GL_TEXTURE_BUFFER, data.size(), &data[0], GL_STREAM_DRAW);
		}
	}

	void DecodeImage()
	{
		int64_t time1, time2;
		GenRGB565();
		glBindImageTexture(0, enc_img, 0, false, 0, GL_READ_ONLY, GL_RGBA16UI);
		glUseProgram(mpgm);
		glUniform2f(uRect, m_w, m_h);

		m_timer.BeginTimer();
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dec_frame);
		glViewport(0, 0, m_w, m_h);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		m_timer.EndTimer();

		time1 = CPUTimer::GetTime();
			DecodeOnCPU(&cpudata[0], &data[0], m_w, m_h, m_type);
		time2 = CPUTimer::GetTime();

		uint64_t time = m_timer.GetTime();

		num_times++;
		totaltime_gpu += time;
		totaltime_cpu += (time2 - time1);
		uint64_t total_avg = m_avgtime.End();

		if (total_avg >= (1000 * 1000))
		{
			printf("Compute shader took: %ldus(%ldms) GPU time (%ldus(%ldms) CPU time) %ld runs in %ldms\n",
				(totaltime_gpu / num_times) / 1000, (totaltime_gpu / num_times) / 1000 / 1000,
				(totaltime_cpu / num_times), (totaltime_cpu / num_times) / 1000,
				num_times, total_avg / 1000);

			num_times = 0;
			totaltime_gpu = totaltime_cpu = 0;
			m_avgtime.Start();
		}
	}

	GLuint GetEncImg() const { return enc_img; }
	GLuint GetDecImg() const { return dec_img; }

private:
	GLuint enc_img, dec_img;
	GLuint enc_buf;

	// Program
	GLuint mpgm;

	// Uniforms
	GLuint uRect;

	GLuint dec_frame;
	TexType m_type;
	int m_w, m_h;
	std::vector<uint8_t> data;
	std::vector<uint32_t> cpudata;
	uint32_t m_shift_val = 1;
	GPUTimer m_timer;
	CPUTimer m_cputime;

	// Average time spent in shader
	CPUTimer m_avgtime;
	uint64_t totaltime_gpu = 0, totaltime_cpu = 0, num_times = 0;
};

TextureConvert* conv;

void DrawTriangle()
{
	conv = new TextureConvert(TexType::TYPE_RGB565, TexDim, TexDim);

	const char* fs_test =
	"#version 310 es\n"
	"precision highp float;\n\n"

	"in vec4 vert;\n"

	"layout(binding = 0) uniform sampler2D tex;\n"

	"out vec4 ocol;\n"
	"void main() {\n"
		"\tvec2 fcoords = vec2(255);\n"
		"\tfcoords = (gl_FragCoord.xy);\n"
		"\tivec2 coords = ivec2(fcoords);\n"
		"\tvec4 out_col = texture(tex, fcoords);\n"
		"\tocol = vec4(out_col) / 255.0;\n"
	"}\n";

	const char* vs_test =
	"#version 310 es\n"

	"in vec4 pos;\n"
	"out vec4 vert;\n"

	"void main() {\n"
		"\tgl_Position = pos;\n"
		"\tvert = pos;\n"
	"}\n";

	GLuint fs, vs, pgm;
	GLint stat, attr_pos, attr_tex;
	fs = glCreateShader(GL_FRAGMENT_SHADER);
	vs = glCreateShader(GL_VERTEX_SHADER);
	pgm = glCreateProgram();

	glShaderSource(fs, 1, &fs_test, NULL);
	glShaderSource(vs, 1, &vs_test, NULL);

	glCompileShader(fs);
	glCompileShader(vs);

	GLUtils::CheckShaderStatus(fs, "fs", fs_test);
	GLUtils::CheckShaderStatus(vs, "vs", vs_test);

	glAttachShader(pgm, fs);
	glAttachShader(pgm, vs);
	glLinkProgram(pgm);

	GLUtils::CheckProgramLinkStatus(pgm);

	glUseProgram(pgm);

	// Get attribute locations
	attr_pos = glGetAttribLocation(pgm, "pos");

	glEnableVertexAttribArray(attr_pos);

	glClearColor(0.4, 0.4, 0.4, 0.0);

	const GLfloat verts[] = {
		-1, -1,
		1, -1,
		-1, 1,
		1, 1,
	};
	const GLfloat tex_coords[] = {
		0.0, 0.0,
		1.0, 0.0,
		0.0, 1.0,
		1.0, 1.0,
	};
	const GLfloat colors[] = {
		1, 0, 0, 1,
		0, 1, 0, 1,
		0, 0, 1, 1,
		1, 1, 0, 1,
	};

	conv->DecodeImage();

	std::clock_t begin, start, end;
	int iters = 0;
	begin = std::clock();
	for (;;)
	{
		conv->DecodeImage();
		glUseProgram(pgm);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//glActiveTexture(GL_TEXTURE0 + 0);
		//glBindTexture(GL_TEXTURE_2D, conv->GetDecImg());
		glVertexAttribPointer(attr_pos, 2, GL_FLOAT, GL_FALSE, 0, verts);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		Context::Swap();
		end = std::clock();
		iters++;
		double duration = (end - begin) / (double)CLOCKS_PER_SEC;
		if (duration >= 1.0)
		{
			printf("iterated: %d\n", iters);
			iters = 0;
			begin = std::clock();
		}

	}
}

static void APIENTRY ErrorCallback( GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char* message, const void* userParam)
{
	printf("Message: '%s'\n", message);
	if (type == GL_DEBUG_TYPE_ERROR_ARB)
		__builtin_trap();
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: %s <tex dim>\n", argv[0]);
		return 0 ;
	}
	TexDim = atoi(argv[1]);
	Context::Create();

	printf("Are we in desktop GL? %s\n", epoxy_is_desktop_gl() ? "Yes" : "No");
	printf("Our GL version %d\n", epoxy_gl_version());

	printf("GL_RENDERER   = %s\n", (char *) glGetString(GL_RENDERER));
	printf("GL_VERSION    = %s\n", (char *) glGetString(GL_VERSION));
	printf("GL_VENDOR     = %s\n", (char *) glGetString(GL_VENDOR));
	printf("GL_EXTENSIONS = %s\n", (char *) glGetString(GL_EXTENSIONS));

	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, true);
	glDebugMessageCallback(ErrorCallback, nullptr);
	glEnable(GL_DEBUG_OUTPUT);

	DrawTriangle();

	Context::Shutdown();
}


use std::time::{Duration, SystemTime};
use std::thread::sleep;

#[macro_use]
extern crate glium;
extern crate rand;

#[allow(unused_imports)]
use glium::{glutin, Surface};

const FRAME_RATE : u32 = 60;
const LOOPBACK_WIDTH : f32 = 2.0;
const JUMP_STRENGTH : f32 = 1.5;
const BLOCK_WIDTH : f32 = 0.1;
const BLOCK_GAP : f32 = 0.3;
const ROT_MULT : f32 = 20f32;
const GRAVITY_STRENGTH : f32 = 3f32;
const START_OFFSET : f32 = -LOOPBACK_WIDTH;
const BRID_LENGTH : f32 = 0.12;
const BRID_WIDTH : f32 = 0.08;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    pub fn get_x(&self) -> f32 {
        self.position[0]
    }

    pub fn get_y(&self) -> f32 {
        self.position[1]
    }
}

struct Brid {
    height: f32,
    angle: f32,
    target_angle: f32,
    velocity: f32,
    is_dead: bool,

    shape: Vec<Vertex>,
    vb: glium::VertexBuffer<Vertex>,
    ib: glium::index::NoIndices,
    shader: glium::Program,
}

impl Brid {

    pub fn new(display: &glium::Display) -> Brid {
        let length = BRID_LENGTH;
        let width = BRID_WIDTH;
        let vertex1 = Vertex { position: [-length, -width] };
        let vertex2 = Vertex { position: [-length,  width] };
        let vertex3 = Vertex { position: [ length,  0.00] };
        let shape = vec![vertex1, vertex2, vertex3];

        let vertbuff = glium::VertexBuffer::new(display, &shape).unwrap();
        let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

        let vertex_shader_src = r#"
            #version 140

            in vec2 position;

            uniform mat4 matrix;

            void main() {
                gl_Position = matrix * vec4(position, 0.0, 1.0);
            }
        "#;

        let fragment_shader_src = r#"
            #version 140

            out vec4 color;

            uniform float col;

            void main() {
                color = vec4(1.0, col, 0.0, 1.0);
            }
        "#;

        let program = glium::Program::from_source(display, vertex_shader_src, fragment_shader_src, None).unwrap();

        Brid {height: 0.0,
        angle: 0.0,
        target_angle: 0.0,
        velocity: 0.0,
        is_dead: false,

        shape: shape,
        vb: vertbuff,
        ib: indices,
        shader: program,
        }
    }

    pub fn draw(&self,  target: &mut glium::Frame) {
        let col: f32;
        if self.is_dead{
            col = 0.0;
        } else {
            col = 1.0;
        }
        let uniforms = uniform! {
            matrix: self.matrix(),
            col: col
        };

        target.draw(&self.vb, &self.ib, &self.shader, &uniforms,
                    &Default::default()).unwrap();
    }

    pub fn update (&mut self, dt: u32, has_jumped: &bool, is_dead: &bool ){
        let d_t: f32 = (dt as f32)/(1_000_000_000.0);
        self.is_dead = *is_dead;

        if !*is_dead{
            if *has_jumped {
                self.target_angle = -0.523598776;
                //self.velocity = JUMP_STRENGTH;
                self.velocity += JUMP_STRENGTH;
                self.update_rotation(d_t);
            } else {
                self.velocity += d_t * -GRAVITY_STRENGTH;
                self.target_angle = -0.523598776 * self.velocity;
                self.update_position(d_t);
                self.update_rotation(d_t);
            }
        } else {
            self.velocity += d_t * -GRAVITY_STRENGTH;
            self.target_angle = -0.523598776 * self.velocity;
            self.update_position(d_t);
            self.update_rotation(d_t);
        }
    }

    fn update_position (&mut self, d_t: f32) {
        self.height += self.velocity * d_t;
        if self.height > 1.0 {
            self.height = 1.0;
            self.velocity = 0.0;
        } else if self.height < -1.0 {
            self.height = -1.0;
            self.velocity = 0.0;
        }
    }

    fn update_rotation (&mut self, d_t: f32) {
        let rot_mult = ROT_MULT;
        if self.angle > self.target_angle {
            self.angle -= rot_mult*d_t;
            if self.angle < self.target_angle {
                self.angle = self.target_angle;
            }
        } else {            
            self.angle += rot_mult*d_t;
            if self.angle > self.target_angle {
                self.angle = self.target_angle;
            }
        }
    }

    pub fn matrix (&self) -> [[f32; 4] ; 4] {
        [
            [ self.angle.cos(), -self.angle.sin(), 0.0, 0.0],
            [ self.angle.sin(),  self.angle.cos(), 0.0, 0.0],
            [ 0.0             ,  0.0             , 1.0, 0.0],
            [-0.6             ,  self.height     , 0.0, 1.0f32]
        ]
    }

    pub fn reset(&mut self){
        self.height = 0.0;
        self.angle = 0.0;
        self.target_angle = 0.0;
        self.velocity = 0.0;
        self.is_dead = false;
    }
    
    pub fn is_collided(&self, bl1: &Block, bl2: &Block, bl3: &Block, bl4: &Block) -> bool {
        let mut is_collided = false;

        for i in 0..3 {
            let point = self.shape[i];
            let x = point.get_x();
            let y = point.get_y();

            let xp = x*self.angle.cos() + y*self.angle.sin() - 0.6;
            let yp = y*self.angle.cos() - x*self.angle.sin() + self.height;
            is_collided = is_collided || bl1.is_collided(&xp,&yp);
            is_collided = is_collided || bl2.is_collided(&xp,&yp);
            is_collided = is_collided || bl3.is_collided(&xp,&yp);
            is_collided = is_collided || bl4.is_collided(&xp,&yp);
        }

        is_collided
    }

    pub fn is_on_ground(&self) -> bool {
        self.height <= -1.0
    }

    pub fn is_about_to_score(&self, bl1: &Block, bl2: &Block, bl3: &Block, bl4: &Block) -> bool {
        let mut is_collided = false;

        let x = -BRID_LENGTH;
        let y = 0.0;

        let xp = x*self.angle.cos() + y*self.angle.sin() - 0.6;
        let yp = y*self.angle.cos() - x*self.angle.sin() + self.height;
        is_collided = is_collided || bl1.inside(&xp,&yp);
        is_collided = is_collided || bl2.inside(&xp,&yp);
        is_collided = is_collided || bl3.inside(&xp,&yp);
        is_collided = is_collided || bl4.inside(&xp,&yp);

        is_collided
    }

}

struct Block {
    gap_centre: f32,
    x_pos: f32,

    vb: glium::VertexBuffer<Vertex>,
    ib: glium::index::NoIndices,
    shader: glium::Program,
}

impl Block {
    pub fn new(x_pos: f32, display: &glium::Display) -> Block {
        let width: f32 = BLOCK_WIDTH;
        let gap: f32 = BLOCK_GAP;
        let vertex1 = Vertex { position: [ width,  gap] };
        let vertex2 = Vertex { position: [ width,  2.00] };
        let vertex3 = Vertex { position: [-width,  gap] };
        let vertex4 = Vertex { position: [-width,  gap] };
        let vertex5 = Vertex { position: [ width,  2.00] };
        let vertex6 = Vertex { position: [-width,  2.00] };

        let vertex7 = Vertex { position: [ width, -gap] };
        let vertex8 = Vertex { position: [-width, -gap] };
        let vertex9 = Vertex { position: [-width, -2.00] };
        let vertex10 = Vertex { position: [ width, -gap] };
        let vertex11 = Vertex { position: [-width, -2.00] };
        let vertex12 = Vertex { position: [ width, -2.00] };
        let shape = vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6,
                         vertex7, vertex8, vertex9, vertex10, vertex11, vertex12];

        let vertbuff = glium::VertexBuffer::new(display, &shape).unwrap();
        let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

        let vertex_shader_src = r#"
            #version 140

            in vec2 position;

            uniform mat4 matrix;

            void main() {
                gl_Position = matrix * vec4(position, 0.0, 1.0);
            }
        "#;

        let fragment_shader_src = r#"
            #version 140

            out vec4 color;

            void main() {
                color = vec4(1.0, 0.1, 0.1, 1.0);
            }
        "#;

        let program = glium::Program::from_source(display, vertex_shader_src, fragment_shader_src, None).unwrap();

        Block {
            gap_centre: 1.4*rand::random::<f32>()-0.7,
            x_pos: x_pos,

            vb: vertbuff,
            ib: indices,
            shader: program,
        }
    }

    fn loopback(&mut self){
        self.gap_centre = 1.4*rand::random::<f32>()-0.7;
        self.x_pos = LOOPBACK_WIDTH;
    }

    pub fn update(&mut self, dt: u32, is_dead: &bool){
        if !*is_dead {
            let d_t: f32 = (dt as f32)/(1_000_000_000.0);
            self.x_pos -= 0.5 * d_t;
            if self.x_pos < -LOOPBACK_WIDTH {
                self.loopback();
            }
        }
    }

    pub fn matrix (&self) -> [[f32; 4] ; 4] {
        [
            [ 1.0        , 0.0, 0.0, 0.0],
            [ 0.0        , 1.0, 0.0, 0.0],
            [ 0.0        , 0.0             , 1.0, 0.0],
            [ self.x_pos , self.gap_centre , 0.0, 1.0f32]
        ]
    }

    pub fn draw(&self,  target: &mut glium::Frame) {
        let uniforms = uniform! {
            matrix: self.matrix()
        };

        target.draw(&self.vb, &self.ib, &self.shader, &uniforms,
                    &Default::default()).unwrap();
    }

    pub fn reset(&mut self, x_pos: f32){
        self.gap_centre = 1.4*rand::random::<f32>()-0.7;
        self.x_pos = x_pos;
    }

    pub fn is_collided(&self, x: &f32, y: &f32) -> bool {
        (self.x_pos - *x).abs() < BLOCK_WIDTH &&
            (self.gap_centre - *y).abs() > BLOCK_GAP
    }

    pub fn inside(&self, x: &f32, y: &f32) -> bool {
        (self.x_pos - *x).abs() < BLOCK_WIDTH &&
            (self.gap_centre - *y).abs() < BLOCK_GAP
    }
}

fn reset_stuff(player: &mut Brid, bl1: &mut Block, bl2: &mut Block, bl3: &mut Block, bl4: &mut Block) {
    player.reset();
    
    bl1.reset(-LOOPBACK_WIDTH/2.0 - START_OFFSET);
    bl2.reset(0.0 - START_OFFSET);
    bl3.reset(LOOPBACK_WIDTH/2.0 - START_OFFSET);
    bl4.reset(LOOPBACK_WIDTH - START_OFFSET);
}

fn main() {
    let sleep_time: u32 = {
        let slp_start = SystemTime::now();
        let blnk_start = SystemTime::now();
        sleep(Duration::new(0,1_000_000));
        let _blnk_duration = blnk_start.elapsed()
            .expect("SystemTime::elapsed failed!");
        let slp_duration = slp_start.elapsed()
            .expect("SystemTime::elapsed failed!");
        slp_duration.subsec_nanos() - 1_000_000
    };

    let mut events_loop = glutin::EventsLoop::new();
    let wb = glutin::WindowBuilder::new()
        .with_title("Crappy Brid - (Press Space) : Score - 0")
        .with_dimensions(glutin::dpi::LogicalSize::new(720.0,720.0));
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();

    implement_vertex!(Vertex, position);

    let frame_time = 1_000_000_000/FRAME_RATE;
    let frame_duration = Duration::new(0, frame_time) - Duration::new(0,sleep_time);

    let mut closed = false;
    let mut jump = false;
    let mut has_jumped = false;
    let mut space_released = true;
    let mut is_dead = false;
    let mut reset = false;
    let mut has_scored = false;
    let mut about_to_score = false;

    let mut score = 0u32;

    let mut player = Brid::new(&display);
    let mut obst1 = Block::new(-LOOPBACK_WIDTH/2.0 - START_OFFSET,&display);
    let mut obst2 = Block::new(0.0 - START_OFFSET,&display);
    let mut obst3 = Block::new(LOOPBACK_WIDTH/2.0 - START_OFFSET,&display);
    let mut obst4 = Block::new(LOOPBACK_WIDTH - START_OFFSET,&display);

    let mut frame_start = SystemTime::now(); 
    let mut phys_start = SystemTime::now();
    while !closed {

        let phys_duration = phys_start.elapsed()
            .expect("SystemTime::elapsed failed!");
        
        phys_start = SystemTime::now();

        if reset {
            about_to_score = false;
            has_scored = false;
            has_jumped = false;
            jump = false;
            score = 0u32;
            reset = false;
            reset_stuff(&mut player, &mut obst1, &mut obst2, &mut obst3, &mut obst4);

            display.gl_window().window().set_title("Crappy Brid - (Press Space) : Score - 0");
        }

        about_to_score = about_to_score || player.is_about_to_score(&mut obst1, &mut obst2, &mut obst3, &mut obst4);

        has_scored = has_scored || about_to_score && !player.is_about_to_score(&mut obst1, &mut obst2, &mut obst3, &mut obst4) && !is_dead;

        if has_scored {
            about_to_score = false;
            has_scored = false;
            score += 1u32;

            let title = format!("{}{}","Crappy Brid - (Press Space) : Score - ",score);
            display.gl_window().window().set_title(&title);
        }

        if has_jumped {
            is_dead = is_dead || player.is_collided(&mut obst1, &mut obst2, &mut obst3, &mut obst4);

            player.update(phys_duration.subsec_nanos() , &jump, &is_dead);
            jump = false;
            obst1.update(phys_duration.subsec_nanos(), &is_dead);
            obst2.update(phys_duration.subsec_nanos(), &is_dead);
            obst3.update(phys_duration.subsec_nanos(), &is_dead);
            obst4.update(phys_duration.subsec_nanos(), &is_dead);
        }

        let mut target = display.draw();
        target.clear_color(0.5, 1.0, 1.0, 1.0);

        player.draw(&mut target);
        obst1.draw(&mut target);
        obst2.draw(&mut target);
        obst3.draw(&mut target);
        obst4.draw(&mut target);

        target.finish().unwrap();

        events_loop.poll_events(|event| {
            match event {
                glutin::Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => closed = true,
                    _ => ()
                },
                glutin::Event::DeviceEvent { event, .. } => match event {
                    glutin::DeviceEvent::Key(
                        glutin::KeyboardInput {
                            virtual_keycode: Some(glutin::VirtualKeyCode::Space),
                            state: glutin::ElementState::Pressed,
                            ..
                        }
                    ) => {
                        if space_released && has_jumped {
                            jump = true;
                            has_jumped = true;
                            space_released = false;
                            if is_dead && player.is_on_ground() {
                                is_dead = false;
                                has_jumped = false;
                                reset = true;
                            }
                        } else {
                            has_jumped = true;
                            space_released = false;
                        }
                    },
                    glutin::DeviceEvent::Key(
                        glutin::KeyboardInput {
                            virtual_keycode: Some(glutin::VirtualKeyCode::Space),
                            state: glutin::ElementState::Released,
                            ..
                        }
                    ) => { 
                        space_released = true;
                    },
                    _ => ()
                },
                _ => (),
            }
        });

        let render_duration = frame_start.elapsed()
            .expect("SystemTime::elapsed failed!");

        let wait_duration: Duration;
        if frame_duration > render_duration {
            wait_duration = frame_duration - render_duration;
            //                 println!("{} : {}", render_duration.subsec_nanos(), wait_duration.subsec_nanos());
            sleep(wait_duration);
        }

    
        frame_start = SystemTime::now();
    }
}

